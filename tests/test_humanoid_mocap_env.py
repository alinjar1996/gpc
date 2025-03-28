from gpc.envs import HumanoidMocapEnv


def test_mocap_env() -> None:
    """Test the humanoid mocap env."""
    env = HumanoidMocapEnv(episode_length=50)

    print(env)


if __name__ == "__main__":
    test_mocap_env()
