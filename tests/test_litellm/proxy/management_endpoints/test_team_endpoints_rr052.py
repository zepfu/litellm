"""
Focused RR-052 coverage for team endpoints:
- org-admin-safe /v2/team/list scoping
- /team/new team_member_budget_duration defaults and forwarding
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy._types import LitellmUserRoles, NewTeamRequest, UserAPIKeyAuth
from litellm.proxy.management_endpoints.team_endpoints import (
    _build_team_list_where_conditions,
    list_team_v2,
    new_team,
)


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_can_list_org_teams():
    """
    Org admins may list teams for organizations where they are ORG_ADMIN
    without providing their own user_id, and results are org-scoped.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )
    mock_teams = [
        Mock(
            model_dump=lambda: {
                "team_id": "team_org_1",
                "team_alias": "Org Team",
                "organization_id": "org-1",
            }
        )
    ]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
        return_value=["org-1", "org-2"],
    ) as mock_get_org_ids:
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 1
        assert len(result["teams"]) == 1
        mock_get_org_ids.assert_awaited_once()
        find_kwargs = mock_db.litellm_teamtable.find_many.await_args.kwargs
        assert find_kwargs["where"]["organization_id"] == {"in": ["org-1", "org-2"]}


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_rejects_out_of_scope_organization_id():
    """
    Org admins cannot query teams for an organization they do not administer.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
        return_value=["org-1"],
    ):
        mock_prisma_client.db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id=None,
                organization_id="org-other",
                user_api_key_dict=mock_user_api_key_dict,
                page=1,
                page_size=10,
                status=None,
            )

        assert exc_info.value.status_code == 403
        assert "only view teams within your organizations" in str(
            exc_info.value.detail
        ).lower()


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_allows_in_scope_organization_id():
    """
    Org admins may filter by an organization_id they administer.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )
    mock_teams = [
        Mock(
            model_dump=lambda: {
                "team_id": "team_org_1",
                "team_alias": "Org Team",
                "organization_id": "org-1",
            }
        )
    ]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
        return_value=["org-1", "org-2"],
    ):
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id="org-1",
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 1
        find_kwargs = mock_db.litellm_teamtable.find_many.await_args.kwargs
        assert find_kwargs["where"]["organization_id"] == "org-1"


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_user_id_filter_keeps_org_scope():
    """
    Org-admin queries with user_id still cannot escape the caller's org scope.

    Even if the target user is a member of teams in org-1 and org-2, an org
    admin for only org-1 must keep organization_id constrained to org-1 while
    also applying the membership/team_id filter.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )
    mock_teams = [
        Mock(
            model_dump=lambda: {
                "team_id": "team-org1",
                "team_alias": "Org1 Team",
                "organization_id": "org-1",
            }
        )
    ]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
        return_value=["org-1"],
    ):
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        target_user = Mock()
        target_user.model_dump.return_value = {
            "user_id": "target-user",
            "teams": ["team-org1", "team-org2"],
        }
        mock_db.litellm_usertable.find_unique = AsyncMock(return_value=target_user)
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)

        result = await list_team_v2(
            http_request=mock_request,
            user_id="target-user",
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 1
        find_kwargs = mock_db.litellm_teamtable.find_many.await_args.kwargs
        where = find_kwargs["where"]
        assert where["organization_id"] == {"in": ["org-1"]}
        assert where["team_id"] == {"in": ["team-org1", "team-org2"]}
        # Cross-org leakage would only be possible if org scope were dropped.
        assert where["organization_id"] != {"in": ["org-1", "org-2"]}


@pytest.mark.asyncio
async def test_build_team_list_where_conditions_org_admin_plus_user_id_combines_filters():
    """
    Direct unit coverage for the Prisma where-clause composition used by
    org-admin + user_id queries.
    """
    prisma_client = Mock()
    user_object = Mock()
    user_object.model_dump.return_value = {
        "user_id": "target-user",
        "teams": ["team-org1", "team-org2"],
    }
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=user_object)

    where = await _build_team_list_where_conditions(
        prisma_client=prisma_client,
        team_id=None,
        team_alias=None,
        organization_id=None,
        user_id="target-user",
        use_deleted_table=False,
        org_admin_org_ids=["org-1"],
    )

    assert where["organization_id"] == {"in": ["org-1"]}
    assert where["team_id"] == {"in": ["team-org1", "team-org2"]}


@pytest.mark.asyncio
async def test_list_team_v2_proxy_admin_view_only_can_list_all_teams():
    """
    PROXY_ADMIN_VIEW_ONLY retains the ability to list all teams.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        user_id="view_only_admin",
    )
    mock_teams = [
        Mock(model_dump=lambda: {"team_id": "team_1", "team_alias": "Team 1"}),
    ]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
    ) as mock_get_org_ids:
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 1
        mock_get_org_ids.assert_not_called()
        find_kwargs = mock_db.litellm_teamtable.find_many.await_args.kwargs
        assert "organization_id" not in find_kwargs["where"]
        assert find_kwargs["where"] == {}


@pytest.mark.asyncio


@pytest.mark.asyncio
async def test_list_team_v2_regular_user_self_scopes_without_user_id():
    """
    Regular users may list their own teams even when user_id is omitted; the
    endpoint auto-scopes the query to the caller.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="regular_user",
    )
    mock_teams = [
        Mock(
            model_dump=lambda: {
                "team_id": "team_self",
                "team_alias": "Self Team",
            }
        )
    ]

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_get_org_ids:
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        self_user = Mock()
        self_user.model_dump.return_value = {
            "user_id": "regular_user",
            "teams": ["team_self"],
        }
        mock_db.litellm_usertable.find_unique = AsyncMock(return_value=self_user)
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 1
        mock_get_org_ids.assert_awaited_once()
        find_unique_kwargs = mock_db.litellm_usertable.find_unique.await_args.kwargs
        assert find_unique_kwargs["where"]["user_id"] == "regular_user"
        find_kwargs = mock_db.litellm_teamtable.find_many.await_args.kwargs
        assert find_kwargs["where"]["team_id"] == {"in": ["team_self"]}
        assert "organization_id" not in find_kwargs["where"]


@pytest.mark.asyncio
async def test_list_team_v2_regular_user_cannot_query_other_user():
    """Regular users cannot list another user's teams."""
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="regular_user",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.user_api_key_cache", Mock()
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", Mock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
        new_callable=AsyncMock,
        return_value=None,
    ):
        mock_prisma_client.db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id="someone_else",
                organization_id=None,
                user_api_key_dict=mock_user_api_key_dict,
                page=1,
                page_size=10,
                status=None,
            )

        assert exc_info.value.status_code == 401


async def test_new_team_forwards_team_member_budget_duration_only():
    """
    Duration-only team member budget requests on /team/new create a budget row
    and forward team_member_budget_duration into the handler.
    """
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
    )
    data = NewTeamRequest(
        team_alias="duration-only-team",
        team_member_budget_duration="30d",
    )

    created_team = MagicMock()
    created_team.team_id = "team_duration_only"
    created_team.model_dump.return_value = {
        "team_id": "team_duration_only",
        "team_alias": "duration-only-team",
        "metadata": {"team_member_budget_id": "budget_duration_only"},
    }

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.llm_router"
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.should_create_budget",
        return_value=True,
    ) as mock_should_create, patch(
        "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.create_team_member_budget_table",
        new_callable=AsyncMock,
    ) as mock_create_budget, patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        new_callable=AsyncMock,
    ):
        mock_prisma_client.get_generic_data = AsyncMock(return_value=None)
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=None
        )
        mock_prisma_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma_client.db.litellm_teamtable.create = AsyncMock(
            return_value=created_team
        )
        mock_prisma_client.jsonify_team_object = MagicMock(
            side_effect=lambda db_data: db_data
        )

        async def _create_side_effect(
            data,
            new_team_data_json,
            user_api_key_dict,
            team_member_budget=None,
            team_member_rpm_limit=None,
            team_member_tpm_limit=None,
            team_member_budget_duration=None,
        ):
            result = dict(new_team_data_json)
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["team_member_budget_id"] = "budget_duration_only"
            result["metadata"] = metadata
            result.pop("team_member_budget_duration", None)
            return result

        mock_create_budget.side_effect = _create_side_effect

        await new_team(
            data=data,
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        mock_should_create.assert_called_once()
        should_kwargs = mock_should_create.call_args.kwargs
        assert should_kwargs["team_member_budget_duration"] == "30d"
        mock_create_budget.assert_awaited_once()
        create_kwargs = mock_create_budget.await_args.kwargs
        assert create_kwargs["team_member_budget_duration"] == "30d"
        assert create_kwargs["team_member_budget"] is None


@pytest.mark.asyncio
async def test_new_team_applies_default_team_member_budget_duration(monkeypatch):
    """
    default_team_params.team_member_budget_duration is applied on /team/new
    when the request omits the field.
    """
    monkeypatch.setattr(
        litellm,
        "default_team_params",
        {"team_member_budget_duration": "7d"},
    )
    monkeypatch.setattr(litellm, "default_team_settings", None)

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
    )
    data = NewTeamRequest(team_alias="default-duration-team")

    created_team = MagicMock()
    created_team.team_id = "team_default_duration"
    created_team.model_dump.return_value = {
        "team_id": "team_default_duration",
        "team_alias": "default-duration-team",
    }

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.llm_router"
    ), patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.should_create_budget",
        return_value=True,
    ) as mock_should_create, patch(
        "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.create_team_member_budget_table",
        new_callable=AsyncMock,
    ) as mock_create_budget, patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        new_callable=AsyncMock,
    ):
        mock_prisma_client.get_generic_data = AsyncMock(return_value=None)
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=None
        )
        mock_prisma_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma_client.db.litellm_teamtable.create = AsyncMock(
            return_value=created_team
        )
        mock_prisma_client.jsonify_team_object = MagicMock(
            side_effect=lambda db_data: db_data
        )

        async def _create_side_effect(
            data,
            new_team_data_json,
            user_api_key_dict,
            team_member_budget=None,
            team_member_rpm_limit=None,
            team_member_tpm_limit=None,
            team_member_budget_duration=None,
        ):
            result = dict(new_team_data_json)
            result.pop("team_member_budget_duration", None)
            return result

        mock_create_budget.side_effect = _create_side_effect

        await new_team(
            data=data,
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        assert data.team_member_budget_duration == "7d"
        assert (
            mock_should_create.call_args.kwargs["team_member_budget_duration"] == "7d"
        )
        assert (
            mock_create_budget.await_args.kwargs["team_member_budget_duration"] == "7d"
        )
