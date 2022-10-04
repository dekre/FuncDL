file(
    name="fundl-base-environment",
    source="environment.yml"
)

python_requirements(
    name="reqs",
    source="3rdparty/requirements.txt",
    module_mapping={
        "scikit-learn": ["sklearn"],
    },
)

python_requirements(
    name="reqs-test",
    source="3rdparty/requirements-test.txt",
)

resource(name="version", source="fundl/.version")

python_sources(
    name="src",
    sources=["fundl/**/*.py"]
)

python_distribution(
    name="dist",
    dependencies=[
        ":src",
        ":version"
        # Dependencies on code to be packaged into the distribution.
    ],
    provides=python_artifact(
        name="fundl",
    ),
    entry_points={
       "console_scripts": {"brawler": "fundl:_cli.app"}
    },
    # Example of setuptools config, other build backends may have other config.
    wheel_config_settings={"--global-option": ["--python-tag", "py39"]},
    # Don't use setuptools with a generated setup.py.
    # You can also turn this off globally in pants.toml:
    #
    # [setup-py-generation]
    # generate_setup_default = false
    wheel=False,
    generate_setup = True,
    repositories=["@codeartifact"]
)

docker_image(
    name="fundl-base-gpu",
    source="docker/Dockerfile.base",
    dependencies=[":fundl-base-environment"],
    repository="dekre/fund-base-gpu",
    image_tags=["py39"]
)

docker_image(
    name="fundl-gpu",
    source="docker/Dockerfile.gpu",
    repository="dekre/fundl-gpu",
    image_tags=["{build_args.PACKAGE_VERSION}"]
)
