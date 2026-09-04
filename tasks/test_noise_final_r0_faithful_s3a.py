from build_noise_final_r0_faithful_s3a_manifest import controls_prefix, tokens_prefix


def main() -> None:
    assert tokens_prefix("4,2,9,7", 3) == "4,2,9"
    assert controls_prefix("1,3,5;2,6,8", 2) == "1,3;2,6"
    try:
        tokens_prefix("1,2", 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("short trajectory did not fail closed")
    print("[test_noise_final_r0_faithful_s3a] PASS")


if __name__ == "__main__":
    main()
