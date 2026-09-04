from build_noise_final_e4_target_cache import REPRESENTATIVE_CELLS, parse_tokens


def main() -> None:
    assert len(REPRESENTATIVE_CELLS) == 8
    assert set(REPRESENTATIVE_CELLS.values()) == {
        "candidate_gradient", "acquisition_positive_gradient", "role_confounder",
    }
    assert parse_tokens("1,4,9").tolist() == [1, 4, 9]
    print("[test_noise_final_e4_target_cache] PASS")


if __name__ == "__main__":
    main()
