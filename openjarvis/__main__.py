"""Entry point: `python -m openjarvis`."""

import asyncio

import click


@click.command()
@click.option("--config", default="config/config.yaml", help="Path to config file.")
def main(config: str) -> None:
    """Launch the OpenJarvis voice assistant."""
    click.echo(f"OpenJarvis starting with config: {config}")
    click.echo("(MVP scaffold — runtime not yet implemented)")
    # TODO: bootstrap event bus, spawn coroutines
    # asyncio.run(bootstrap(config))


if __name__ == "__main__":
    main()
