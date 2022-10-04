from typer import Typer, Option
from dotenv import load_dotenv

app = Typer()


@app.command()
def fit(
    env_file: str = Option(".env"),
    log_level: str = Option("info"),
):
    load_dotenv(env_file)


@app.command()
def evaluate(
    env_file: str = Option(".env"),
    log_level: str = Option("info"),
):
    load_dotenv(env_file)


if __name__ == "__main__":
    app()
