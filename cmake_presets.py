import os
import json
import argparse


def detect_conda_env():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is None:
        raise RuntimeError("No active conda environment detected. Please activate one or specify --env-name.")
    return conda_prefix


def generate_presets(env_path, env_name, build_type="Debug", preset_name="conda-debug"):
    presets = {
        "version": 3,
        "cmakeMinimumRequired": {"major": 3, "minor": 22},
        "configurePresets": [
            {
                "name": "default",
                "hidden": True,
                "generator": "Ninja",
                "binaryDir": "${sourceDir}/build/${presetName}",
                "cacheVariables": {"CMAKE_EXPORT_COMPILE_COMMANDS": "YES"},
            },
            {
                "name": preset_name,
                "inherits": "default",
                "description": f"Configure using conda env {env_name}",
                "cacheVariables": {"CMAKE_BUILD_TYPE": build_type, "CONDA_ENV_PATH": f"{env_path}"},
                "environment": {"PATH": f"{env_path}/bin:$penv" + "{PATH}"},
            },
        ],
        "buildPresets": [{"name": preset_name, "configurePreset": preset_name}],
    }
    return presets


def write_presets(presets, output_path):
    with open(output_path, "w") as f:
        json.dump(presets, f, indent=2)
    print(f"[✓] CMakePresets.json written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-generate CMakePresets.json for a conda environment.")
    parser.add_argument("--env-name", type=str, help="Conda environment name (optional if already activated)")
    parser.add_argument("--env-path", type=str, help="Full path to conda env (optional if already activated)")
    parser.add_argument("--build-type", type=str, default="Debug", choices=["Debug", "Release"])
    parser.add_argument("--preset-name", type=str, default="conda-debug")
    parser.add_argument("--output", type=str, default="CMakePresets.json")

    args = parser.parse_args()

    # Detect or resolve conda env path
    if args.env_path:
        env_path = args.env_path
    elif args.env_name:
        env_path = f"$HOME/anaconda3/envs/{args.env_name}"
    else:
        env_path = detect_conda_env()

    # Try to infer env name if not provided
    env_name = args.env_name or os.path.basename(env_path)

    # Generate and write
    presets = generate_presets(env_path, env_name, build_type=args.build_type, preset_name=args.preset_name)
    write_presets(presets, args.output)


if __name__ == "__main__":
    main()
