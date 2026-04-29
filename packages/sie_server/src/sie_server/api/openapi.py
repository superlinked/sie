"""Custom OpenAPI schema generation for SIE Server.

Provides dynamic OpenAPI schema customization, such as setting path parameter
examples based on registered models.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from sie_server.types.openapi import EncodeRequestModel, ExtractRequestModel, ScoreRequestModel


def setup_custom_openapi_schema(app: FastAPI) -> None:
    """Configure custom OpenAPI schema generation for the app.

    This customizes the OpenAPI schema to use the first registered model
    as the example value for all `model` path parameters.
    """

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        _add_request_body_schemas(openapi_schema)

        # Set model path parameter examples to first registered model
        registry = getattr(app.state, "registry", None)
        if registry and registry.model_names:
            first_model = registry.model_names[0]
            _set_model_examples(openapi_schema, first_model)

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore


def _add_request_body_schemas(openapi_schema: dict[str, Any]) -> None:
    """Add request body model schemas to OpenAPI components."""
    # Ensure components.schemas exists
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    schemas = openapi_schema["components"]["schemas"]

    # Add request model schemas using Pydantic's model_json_schema
    for model_class in [EncodeRequestModel, ExtractRequestModel, ScoreRequestModel]:
        model_name = model_class.__name__
        if model_name not in schemas:
            # Get the full schema including $defs for nested models
            full_schema = model_class.model_json_schema()
            # Extract and add any nested definitions
            if "$defs" in full_schema:
                for def_name, def_schema in full_schema["$defs"].items():
                    if def_name not in schemas:
                        schemas[def_name] = def_schema
                del full_schema["$defs"]
            schemas[model_name] = full_schema


def _set_model_examples(openapi_schema: dict[str, Any], model_name: str) -> None:
    """Update all 'model' path parameters with the given example."""
    for path_data in openapi_schema.get("paths", {}).values():
        for method_data in path_data.values():
            if not isinstance(method_data, dict):
                continue
            for param in method_data.get("parameters", []):
                if param.get("name") == "model" and param.get("in") == "path":
                    param["example"] = model_name
