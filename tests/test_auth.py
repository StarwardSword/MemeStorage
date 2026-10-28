import sqlite3
import tempfile
import time
import unittest
import uuid
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from api.http.config import Configuration as HttpConfiguration
from api.http.serve import create_app
from app import (
    ActivityConfiguration,
    ActivityRecorder,
    Event,
    StoredFile,
    UserExistsError,
    UserNotFoundError,
    outcome_of,
)
from infrastructure.activity import SqliteActivityStore
from infrastructure.file_storage import Configuration as FilesConfiguration
from infrastructure.file_storage import LocalFileStorage

from .fakes import ADMIN_TOKEN, make_app, message


def local_files(directory):
    return LocalFileStorage(
        FilesConfiguration(
            root=str(Path(directory, "files")),
            thumbnails_root=str(Path(directory, "thumbnails")),
            sqlite_connection_string=str(Path(directory, "app.db")),
        )
    )


class UsersTest(unittest.TestCase):
    def test_tokens_open_the_door_until_the_user_is_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            user, token = app.create_user("  anna  ", allow_embeddings=False)
            self.assertEqual(user.name, "anna")

            principal = app.authenticate(token)
            self.assertEqual((principal.user_id, principal.is_admin), (user.id, False))
            for bad in (None, "", "mst_nope", token + "x", token[:-1]):
                self.assertIsNone(app.authenticate(bad))

            app.delete_user(user.id)
            self.assertIsNone(app.authenticate(token))
            # The row stays so that files and statistics keep their author
            self.assertTrue(app.list_users()[0].deleted_at)
            with self.assertRaises(UserNotFoundError):
                app.delete_user(user.id)

    def test_names_are_unique_even_after_deletion_and_checked(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            user, _ = app.create_user("anna", False)
            with self.assertRaises(UserExistsError):
                app.create_user("anna", False)
            app.delete_user(user.id)
            with self.assertRaises(UserExistsError):
                app.create_user("anna", False)
            for bad in ("", "   ", "a" * 65, "bad/name", "tab\there"):
                with self.assertRaises(ValueError):
                    app.create_user(bad, False)

    def test_a_new_token_replaces_the_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            user, old = app.create_user("anna", False)
            new, restored = app.rotate_token(user.id)
            self.assertFalse(restored)
            self.assertIsNone(app.authenticate(old))
            self.assertEqual(app.authenticate(new).user_id, user.id)
            self.assertEqual(app.authenticate(ADMIN_TOKEN).is_admin, True)
            self.assertIsNone(app.authenticate(ADMIN_TOKEN + "x"))

    def test_a_deleted_user_comes_back_with_a_new_token(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            user, old = app.create_user("anna", True)
            app.delete_user(user.id)
            self.assertIsNone(app.authenticate(old))

            token, restored = app.rotate_token(user.id)
            self.assertTrue(restored)
            principal = app.authenticate(token)
            self.assertEqual(
                (principal.user_id, principal.allow_embeddings), (user.id, True)
            )
            self.assertIsNone(app.authenticate(old))
            self.assertIsNone(app.list_users()[0].deleted_at)

            with self.assertRaises(UserNotFoundError):
                app.rotate_token(uuid.uuid7())

    def test_the_door_of_a_deleted_user_stays_shut_without_a_token(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            user, _ = app.create_user("anna", False)
            app.delete_user(user.id)
            # Changing settings of a deleted user is refused, bring it back first
            with self.assertRaises(UserNotFoundError):
                app.update_user(user.id, True)

    def test_tokens_are_not_kept(self):
        with tempfile.TemporaryDirectory() as d:
            app = make_app(d)
            _, token = app.create_user("anna", False)
            with closing(sqlite3.connect(Path(d, "app.db"))) as conn:
                dump = " ".join(conn.iterdump())
            self.assertNotIn(token, dump)


class StatsTest(unittest.TestCase):
    def test_files_are_counted_by_owner_and_type(self):
        with tempfile.TemporaryDirectory() as d:
            files = local_files(d)
            app = make_app(d, files=files)
            anna, _ = app.create_user("anna", False)
            bob, _ = app.create_user("bob", False)

            def put(owner, content_type, size):
                files.save(
                    StoredFile(
                        id=uuid.uuid7(),
                        data=iter([b"x" * size]),
                        content_type=content_type,
                        original_name="f",
                        owner_id=owner,
                    )
                )

            put(anna.id, "image/png", 10)
            put(anna.id, "image/jpeg", 20)
            put(anna.id, "video/mp4", 100)
            put(None, "audio/wav", 5)
            app.delete_user(bob.id)

            stats = app.stats()
            by_name = {u.name: u for u in stats.users}
            self.assertEqual(by_name["anna"].files, 3)
            self.assertEqual(by_name["anna"].bytes, 130)
            self.assertEqual(by_name["anna"].by_type["image"].files, 2)
            self.assertEqual(by_name["anna"].by_type["video"].bytes, 100)
            self.assertEqual((by_name["bob"].files, by_name["bob"].deleted), (0, True))
            self.assertEqual(by_name[None].files, 1)
            self.assertEqual((stats.files, stats.bytes), (4, 135))

            listed = app.list_files(anna.id, 10, 0)
            self.assertEqual(len(listed), 3)
            self.assertTrue(all(f.owner_id == anna.id for f in listed))
            self.assertEqual(len(app.list_files(None, 2, 0)), 2)

    def test_submitted_files_keep_their_author(self):
        with tempfile.TemporaryDirectory() as d:
            files = local_files(d)
            app = make_app(d, files=files)
            anna, _ = app.create_user("anna", False)
            job = app.submit(message("a.png", 50, owner_id=anna.id))
            self.assertEqual(app.get_job(job.id).owner_id, anna.id)
            self.assertEqual(app.list_files(anna.id, 10, 0)[0].size, 50)

    def test_database_made_before_owners_gets_the_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "app.db")
            with closing(sqlite3.connect(path)) as conn:
                conn.execute(
                    "CREATE TABLE file_data(id TEXT PRIMARY KEY NOT NULL CHECK (length(id) = 36),"
                    " content_type TEXT NOT NULL, original_name TEXT NOT NULL)"
                )
                conn.execute(
                    "INSERT INTO file_data VALUES (?, 'image/png', 'old.png')",
                    (str(uuid.uuid7()),),
                )
                conn.commit()
            app = make_app(d, files=local_files(d))
            only = app.stats().users[0]
            self.assertEqual((only.user_id, only.files, only.bytes), (None, 1, 0))


class ActivityTest(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(outcome_of("200"), "ok")
        self.assertEqual(outcome_of("404"), "client_error")
        self.assertEqual(outcome_of("401"), "denied")
        self.assertEqual(outcome_of("500"), "server_error")
        self.assertEqual(outcome_of("OK"), "ok")
        self.assertEqual(outcome_of("PERMISSION_DENIED"), "denied")
        self.assertEqual(outcome_of("INTERNAL"), "server_error")
        self.assertEqual(outcome_of("NOT_FOUND"), "client_error")

    def test_summary_counts_what_was_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            store = SqliteActivityStore(str(Path(d, "app.db")))
            recorder = ActivityRecorder(store, ActivityConfiguration())
            now = time.time()
            anna = uuid.uuid7()

            def event(action, code, ms, user=anna, detail=None, ts=now):
                return Event(
                    ts=ts,
                    transport="http",
                    action=action,
                    code=code,
                    outcome=outcome_of(code),
                    user_id=user,
                    actor="anna" if user else None,
                    duration_ms=ms,
                    bytes_in=10,
                    bytes_out=20,
                    ip="1.2.3.4",
                    detail=detail,
                )

            recorder.start()
            for ms in range(1, 101):
                recorder.record(event("search_files", "200", ms, detail="cat"))
            recorder.record(event("search_files", "200", 5, detail="dog"))
            recorder.record(event("get_file", "404", 3))
            recorder.record(event("save_file", "401", 1, user=None))
            recorder.record(event("old", "200", 1, ts=now - 30 * 86400))
            recorder.stop()

            s = store.summary(now - 86400, now + 60)
            self.assertEqual(
                (s.requests, s.ok, s.client_error, s.denied), (103, 101, 1, 1)
            )
            self.assertEqual((s.bytes_in, s.bytes_out), (1030, 2060))
            searches = next(a for a in s.by_action if a.action == "search_files")
            self.assertEqual(searches.requests, 101)
            self.assertEqual(searches.p95_ms, 95)
            self.assertEqual(searches.max_ms, 100)
            self.assertEqual(
                [(q.query, q.count) for q in s.top_queries], [("cat", 100), ("dog", 1)]
            )
            self.assertEqual(s.by_ip[0].ip, "1.2.3.4")
            self.assertEqual(s.by_ip[0].denied, 1)
            self.assertEqual({a.actor for a in s.by_actor}, {"anna", None})
            self.assertEqual(len(s.by_day), 1)

            only_denied = store.events(
                now - 86400, now + 60, None, None, "denied", 10, 0
            )
            self.assertEqual([e.action for e in only_denied], ["save_file"])
            self.assertEqual(store.prune(now - 7 * 86400), 1)

    def test_queries_can_stay_private(self):
        with tempfile.TemporaryDirectory() as d:
            store = SqliteActivityStore(str(Path(d, "app.db")))
            recorder = ActivityRecorder(
                store, ActivityConfiguration(record_queries=False)
            )
            recorder.start()
            recorder.record(
                Event(
                    ts=time.time(),
                    transport="http",
                    action="search_files",
                    code="200",
                    outcome="ok",
                    detail="secret",
                )
            )
            recorder.stop()
            self.assertIsNone(
                store.events(0, time.time() + 1, None, None, None, 5, 0)[0].detail
            )

    def test_a_full_queue_drops_events_without_waiting(self):
        with tempfile.TemporaryDirectory() as d:
            recorder = ActivityRecorder(
                SqliteActivityStore(str(Path(d, "app.db"))), ActivityConfiguration()
            )
            # The writer is not started, so nothing is taken out
            event = Event(ts=1, transport="http", action="a", code="200", outcome="ok")
            for _ in range(10_000 + 5):
                recorder.record(event)
            self.assertEqual(recorder.dropped, 5)


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.app = make_app(
            self.dir.name, files=local_files(self.dir.name), activity=True
        )
        self.app.activity.start()
        self.client = TestClient(create_app(self.app, HttpConfiguration()))
        self.admin = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

    def tearDown(self):
        self.app.activity.stop()
        self.dir.cleanup()

    def test_everything_but_health_needs_a_token(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        for method, url in (
            ("get", "/files/" + str(uuid.uuid7())),
            ("get", "/search?q=cat"),
            ("get", "/jobs/" + str(uuid.uuid7())),
            ("post", "/jobs"),
            ("get", "/admin/users"),
        ):
            r = getattr(self.client, method)(url)
            self.assertEqual(r.status_code, 401, url)
            self.assertEqual(r.headers["www-authenticate"], "Bearer")
        r = self.client.get(
            "/admin/users", headers={"Authorization": "Bearer mst_wrong"}
        )
        self.assertEqual(r.status_code, 401)

    def test_a_user_token_is_not_an_admin_token(self):
        r = self.client.post("/admin/users", json={"name": "anna"}, headers=self.admin)
        self.assertEqual(r.status_code, 201)
        token = r.json()["token"]
        user = {"Authorization": f"Bearer {token}"}

        self.assertEqual(self.client.get("/admin/users", headers=user).status_code, 403)
        self.assertEqual(self.client.get("/admin/stats", headers=user).status_code, 403)
        # A user token passes the door of the normal routes
        missing = self.client.get(f"/jobs/{uuid.uuid7()}", headers=user)
        self.assertEqual(missing.status_code, 404)

    def test_user_lifecycle_through_the_api(self):
        a = self.admin
        created = self.client.post(
            "/admin/users", json={"name": "anna", "allow_embeddings": True}, headers=a
        )
        user = created.json()["user"]
        self.assertTrue(user["allow_embeddings"])
        self.assertEqual(
            self.client.post(
                "/admin/users", json={"name": "anna"}, headers=a
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.post(
                "/admin/users", json={"name": "a/b"}, headers=a
            ).status_code,
            422,
        )

        patched = self.client.patch(
            f"/admin/users/{user['id']}", json={"allow_embeddings": False}, headers=a
        )
        self.assertFalse(patched.json()["allow_embeddings"])

        token = self.client.post(f"/admin/users/{user['id']}/token", headers=a).json()[
            "token"
        ]
        self.assertNotEqual(token, created.json()["token"])
        self.assertEqual(
            self.client.get(
                f"/jobs/{uuid.uuid7()}",
                headers={"Authorization": f"Bearer {created.json()['token']}"},
            ).status_code,
            401,
        )

        self.assertEqual(
            self.client.delete(f"/admin/users/{user['id']}", headers=a).status_code, 204
        )
        self.assertEqual(
            self.client.delete(f"/admin/users/{user['id']}", headers=a).status_code, 404
        )
        self.assertEqual(
            self.client.get(
                f"/jobs/{uuid.uuid7()}", headers={"Authorization": f"Bearer {token}"}
            ).status_code,
            401,
        )
        self.assertTrue(
            self.client.get("/admin/users", headers=a).json()[0]["deleted_at"]
        )

        back = self.client.post(f"/admin/users/{user['id']}/token", headers=a).json()
        self.assertTrue(back["restored"])
        revived = {"Authorization": f"Bearer {back['token']}"}
        self.assertEqual(
            self.client.get(f"/jobs/{uuid.uuid7()}", headers=revived).status_code, 404
        )
        self.assertIsNone(
            self.client.get("/admin/users", headers=a).json()[0]["deleted_at"]
        )

    def test_every_request_leaves_a_trace(self):
        a = self.admin
        user = self.client.post("/admin/users", json={"name": "anna"}, headers=a).json()
        token = {"Authorization": f"Bearer {user['token']}"}
        self.client.get(f"/jobs/{uuid.uuid7()}", headers=token)
        self.client.get("/jobs/" + str(uuid.uuid7()))
        self.client.get("/health")

        self.app.activity.flush()
        events = self.client.get("/admin/activity/events", headers=a).json()
        seen = [(e["action"], e["code"], e["actor"]) for e in events]
        self.assertIn(("get_job", "404", "anna"), seen)
        self.assertIn(("get_job", "401", None), seen)
        self.assertIn(("create_user", "201", "admin"), seen)
        self.assertNotIn("health", [e["action"] for e in events])
        self.assertTrue(all(e["ip"] for e in events))

        summary = self.client.get("/admin/activity/summary", headers=a).json()
        self.assertGreaterEqual(summary["requests"], 3)
        self.assertGreaterEqual(summary["denied"], 1)
        self.assertEqual(summary["jobs"]["by_status"], {})

        stats = self.client.get("/admin/stats", headers=a).json()
        self.assertEqual([u["name"] for u in stats["users"]], ["anna"])
        self.assertEqual(
            self.client.delete(f"/admin/files/{uuid.uuid7()}", headers=a).status_code,
            404,
        )

    def test_a_search_query_is_logged_with_its_text(self):
        a = self.admin
        token = {
            "Authorization": f"Bearer {self.client.post('/admin/users', json={'name': 'anna'}, headers=a).json()['token']}"
        }
        # No model in this test, so the search itself fails, the trace of it must stay
        with self.assertRaises(Exception):
            self.client.get("/search?q=sad%20cat", headers=token)
        self.app.activity.flush()
        events = self.client.get(
            "/admin/activity/events?action=search_files", headers=a
        ).json()
        self.assertEqual([e["detail"] for e in events], ["sad cat"])


if __name__ == "__main__":
    unittest.main()
