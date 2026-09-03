import unittest

from wmloop.execute.window_scheduler import WindowSchedulerError, build_window_schedule


class WindowSchedulerTests(unittest.TestCase):
    def test_probe_repeats_one_window(self):
        records = [{"episode_id": "a", "horizon_frames": 150}]
        self.assertEqual(build_window_schedule(records, steps=3, mode="probe", chunk_frames=45, sample_index=0), [(0, 0)] * 3)

    def test_episode_balanced_is_order_independent_by_episode(self):
        records = [
            {"episode_id": "a", "horizon_frames": 150},
            {"episode_id": "a", "horizon_frames": 150},
            {"episode_id": "b", "horizon_frames": 150},
            {"episode_id": "c", "horizon_frames": 150},
        ]
        schedule = build_window_schedule(records, steps=6, mode="long", record_limit=3, chunk_frames=45, seed=7)
        episodes = [records[index]["episode_id"] for index, _offset in schedule[:3]]
        self.assertEqual(len(set(episodes)), 3)
        self.assertEqual(schedule, build_window_schedule(records, steps=6, mode="long", record_limit=3, chunk_frames=45, seed=7))

    def test_offset_rotation_includes_tail(self):
        records = [{"episode_id": "a", "horizon_frames": 150}]
        schedule = build_window_schedule(records, steps=8, mode="long", chunk_frames=45)
        self.assertEqual({offset for _index, offset in schedule}, {0, 45, 90, 105})

    def test_invalid_manifest_is_fail_closed(self):
        with self.assertRaisesRegex(WindowSchedulerError, "EPISODE_ID_INVALID"):
            build_window_schedule([{"horizon_frames": 150}], steps=1, chunk_frames=45)


if __name__ == "__main__":
    unittest.main()
