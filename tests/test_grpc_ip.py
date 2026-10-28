import unittest

from api.grpc.activity import client_ip, is_loopback, peer_ip


class Context:
    def __init__(self, peer, forwarded=None):
        self._peer = peer
        self._metadata = [] if forwarded is None else [("x-forwarded-for", forwarded)]

    def peer(self):
        return self._peer

    def invocation_metadata(self):
        return self._metadata


class PeerTest(unittest.TestCase):
    def test_the_address_is_cut_out_of_the_peer_string(self):
        self.assertEqual(peer_ip("ipv4:1.2.3.4:5555"), "1.2.3.4")
        self.assertEqual(peer_ip("ipv6:[2a00::1]:5555"), "2a00::1")
        self.assertEqual(peer_ip("ipv6:[::ffff:127.0.0.1]:5555"), "::ffff:127.0.0.1")
        self.assertIsNone(peer_ip("unix:/run/x.sock"))

    def test_loopback_in_all_its_forms(self):
        for ip in ("127.0.0.1", "127.1.2.3", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(is_loopback(ip), ip)
        for ip in ("1.2.3.4", "2a00::1", "::ffff:1.2.3.4", "", None, "localhost"):
            self.assertFalse(is_loopback(ip), ip)


class ClientIpTest(unittest.TestCase):
    def test_behind_nginx_the_forwarded_address_is_the_client(self):
        self.assertEqual(
            client_ip(Context("ipv4:127.0.0.1:4000", "203.0.113.7")), "203.0.113.7"
        )
        self.assertEqual(
            client_ip(Context("ipv6:[::1]:4000", "2001:db8::5")), "2001:db8::5"
        )
        self.assertEqual(
            client_ip(Context("ipv6:[::ffff:127.0.0.1]:4000", "203.0.113.7")),
            "203.0.113.7",
        )

    def test_only_the_first_of_a_chain_counts(self):
        ctx = Context("ipv4:127.0.0.1:4000", "203.0.113.7, 10.0.0.1")
        self.assertEqual(client_ip(ctx), "203.0.113.7")

    def test_a_stranger_can_not_name_his_own_address(self):
        # Directly connected, so the header is written by the client himself
        self.assertEqual(
            client_ip(Context("ipv4:198.51.100.9:4000", "203.0.113.7")), "198.51.100.9"
        )
        self.assertEqual(
            client_ip(Context("ipv6:[2a00::9]:4000", "203.0.113.7")), "2a00::9"
        )

    def test_without_the_header_or_with_rubbish_the_peer_is_used(self):
        self.assertEqual(client_ip(Context("ipv4:127.0.0.1:4000")), "127.0.0.1")
        for rubbish in (
            "",
            "not an address",
            "1.2.3.4\nERROR forged line",
            "999.1.1.1",
        ):
            self.assertEqual(
                client_ip(Context("ipv4:127.0.0.1:4000", rubbish)),
                "127.0.0.1",
                repr(rubbish),
            )

    def test_a_unix_socket_has_no_address(self):
        self.assertIsNone(client_ip(Context("unix:/run/x.sock", "203.0.113.7")))


if __name__ == "__main__":
    unittest.main()
