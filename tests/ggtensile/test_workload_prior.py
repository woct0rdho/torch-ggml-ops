from tools.ggtensile.workload_prior import (
    EXPERT_PRIOR_NAMES,
    ExpertPrior,
    default_profile_seed,
    expert_prior_metadata,
    sample_expert_profile,
)
from tools.tune_grouped_mmq_prior import generate_document


def test_fitted_profiles_are_deterministic_and_preserve_route_contract() -> None:
    expected = {
        ExpertPrior.QwenLearned: (
            8,
            ((2_048, 233, 1_297), (8_192, 253, 4_192), (32_768, 253, 20_802)),
        ),
        ExpertPrior.DeepSeekLearned: (
            6,
            ((2_048, 244, 982), (8_192, 253, 3_441), (32_768, 254, 12_476)),
        ),
        ExpertPrior.DeepSeekHash: (
            6,
            ((2_048, 256, 264), (8_192, 256, 942), (32_768, 256, 3_765)),
        ),
    }
    for prior, (top_k, sizes) in expected.items():
        for tokens, active, maximum in sizes:
            first = sample_expert_profile(prior, tokens)
            second = sample_expert_profile(prior, tokens)
            assert first == second
            assert first.seed == default_profile_seed(prior, tokens)
            assert first.top_k == top_k
            assert first.aggregate_rows == top_k * tokens
            assert sum(rows > 0 for rows in first.rows_per_expert) == active
            assert max(first.rows_per_expert) == maximum
            assert len(first.rows_per_expert) == 256
            assert all(0 <= rows <= first.tokens for rows in first.rows_per_expert)


def test_profile_document_materializes_only_the_selected_law() -> None:
    for prior in EXPERT_PRIOR_NAMES:
        document = generate_document(prior, (1, 4, 16), 2_048)
        assert document["fit"] == expert_prior_metadata(prior)
        assert document["profile_policy"] == (
            "one deterministic profile per physical token count"
        )
        profiles = document["profiles"]
        assert isinstance(profiles, list)
        assert len(profiles) == 3
        assert {profile["law"] for profile in profiles} == {prior}
        assert all(len(profile["rows_per_expert"]) == 256 for profile in profiles)
        assert all("medoid" not in profile for profile in profiles)
