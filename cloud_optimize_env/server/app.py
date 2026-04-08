"""
FastAPI application for the CloudOptimize Environment.
"""

from openenv.core.env_server.http_server import create_app

from .environment import CloudOptimizeEnvironment

from cloud_optimize_env.models import CloudOptimizeAction, CloudOptimizeObservation

app = create_app(
    CloudOptimizeEnvironment,
    CloudOptimizeAction,
    CloudOptimizeObservation,
    env_name="cloud_optimize_env",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
