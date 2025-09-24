import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from starlette.routing import Match, Mount, Route
from starlette.types import Scope

if TYPE_CHECKING:
    from starlette.requests import Request

LOG = logging.getLogger(__name__)


class RouterPathMatcher:
    @classmethod
    def get_matching_route_path(
        cls, scope: Dict[Any, Any], routes: List[Route], route_name: Optional[str] = None
    ) -> str:
        """
        Find a matching route and return its original path string

        Will attempt to enter mounted routes and subrouters.

        Credit to https://github.com/elastic/apm-agent-python
        """
        for route in routes:
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                route_name = route.path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child_route_name = cls.get_matching_route_path(child_scope, route.routes, route_name)
                    if child_route_name is None:
                        route_name = None
                    else:
                        route_name += child_route_name
                return route_name
            elif match == Match.PARTIAL and route_name is None:
                route_name = route.path

    @classmethod
    def get_router_path_from_request(cls, request: "Request") -> Optional[str]:
        return cls.get_router_path(request.scope)

    @classmethod
    def get_router_path(cls, scope: Scope) -> Optional[str]:
        """Returns the original router path (with url param names) for given request."""
        app = scope.get("app", {})
        if not scope.get("router", None) and not app:
            return None

        root_path = scope.get("root_path", "")
        if hasattr(app, "root_path"):
            app_root_path = getattr(app, "root_path")
            if root_path.startswith(app_root_path):
                root_path = root_path[len(app_root_path) :]

        base_scope = {
            "type": scope.get("type"),
            "path": root_path + scope.get("path"),
            "path_params": scope.get("path_params", {}),
            "method": scope.get("method"),
        }

        try:
            return cls.get_matching_route_path(base_scope, (scope.get("router") or app.router).routes)
        except Exception as ex:
            LOG.warning(f"Path not found; {ex}")

        return None
