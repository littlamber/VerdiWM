import pytest
from wmloop.diagnose.episode_inventory import EpisodeInventoryError, summarize_episodes


def record(uid='a', split='val', stream='front', frames=300, fps=5, start=4):
    return dict(episode_uid=uid,split=split,stream_id=stream,num_frames=frames,fps=fps,first_future_frame=start)


def test_forecast_clock_excludes_observed_prefix_and_counts_episodes():
    report = summarize_episodes([record(), record(stream='side',frames=304), record(uid='b',frames=450)],horizons_seconds=[60,90])
    val = report['splits']['val']
    assert val['stream_count'] == 3
    assert val['unique_episode_count'] == 2
    assert val['horizons'][0] == dict(seconds=60,recording_stream_count=3,forecast_stream_count=2,forecast_unique_episode_count=2)
    assert val['horizons'][1]['forecast_unique_episode_count'] == 0


def test_shared_episode_across_views_and_splits_is_leakage():
    report=summarize_episodes([record(split='train'),record(stream='side')],horizons_seconds=[60])
    assert report['state']=='blocked'
    assert report['split_overlaps'][0]['episode_uids']==['a']


def test_short_episodes_remain_in_inventory_with_zero_future_support():
    report=summarize_episodes([record(frames=1)],horizons_seconds=[1])
    assert report['records'][0]['forecast_seconds']==0
    assert report['splits']['val']['horizons'][0]['forecast_stream_count']==0


@pytest.mark.parametrize('changes', [{'fps':0},{'fps':float('nan')},{'num_frames':True},{'episode_uid':''}])
def test_invalid_metadata_is_rejected(changes):
    with pytest.raises(EpisodeInventoryError):
        summarize_episodes([record()|changes],horizons_seconds=[60])


def test_duplicate_stream_cannot_inflate_support():
    with pytest.raises(EpisodeInventoryError,match='DUPLICATE'):
        summarize_episodes([record(),record()],horizons_seconds=[60])
