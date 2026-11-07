"""Dash/Plotly dashboard for SEFS visualization, conflict resolution, and timeline views."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from werkzeug.serving import make_server

import config


class SEFSDashboard:
    """Serve and maintain the live SEFS dashboard with map and timeline tabs."""

    def __init__(
        self,
        snapshot_provider: Callable[[], dict[str, Any]],
        query_handler: Callable[[str, int], dict[str, Any]] | None = None,
        override_handler: Callable[[str, str | None], tuple[bool, str]] | None = None,
        rename_handler: Callable[[str, str], None] | None = None,
        delete_handler: Callable[[str], None] | None = None,
        host: str = config.DASHBOARD_HOST,
        port: int = config.DASHBOARD_PORT,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.query_handler = query_handler
        self.override_handler = override_handler
        self.rename_handler = rename_handler
        self.delete_handler = delete_handler
        self.host = host
        self.port = port

        self.logger = logging.getLogger(self.__class__.__name__)
        self._server_lock = threading.Lock()
        self._server = None
        self._thread: threading.Thread | None = None

        self.app = Dash(__name__, title="SEFS | Smart File Workspace")
        self.app.layout = self._build_layout()
        self._register_callbacks()

    def start(self) -> None:
        """Start the dashboard server in a background thread."""
        with self._server_lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._serve, name="sefs-dashboard", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard server and wait for thread termination."""
        with self._server_lock:
            server = self._server

        if server is not None:
            server.shutdown()

        with self._server_lock:
            thread = self._thread

        if thread and thread.is_alive():
            thread.join(timeout=5)

    def _serve(self) -> None:
        server = make_server(self.host, self.port, self.app.server, threaded=True)
        with self._server_lock:
            self._server = server

        self.logger.info("Dashboard running at http://%s:%s", self.host, self.port)
        try:
            server.serve_forever()
        finally:
            with self._server_lock:
                self._server = None

    def _build_layout(self) -> html.Div:
        return html.Div(
            children=[
                dcc.Store(id="snapshot-store", data={}),
                dcc.Store(id="selected-file-path", data=""),
                html.Header(
                    [
                        html.Div("SEFS", className="hero-eyebrow"),
                        html.H1("Your files, understood."),
                        html.P(
                            "A live semantic workspace for discovering, organizing, and exploring your documents.",
                            className="hero-description",
                        ),
                    ],
                    className="hero",
                ),
                html.Div(id="stats-panel", className="stats-region"),
                html.Div(
                    children=[
                        html.Div(
                            [
                                html.Div("01 / ASK YOUR FILES", className="section-eyebrow"),
                                html.H2("Semantic search", className="section-title"),
                                html.P(
                                    "Ask a question in your own words and find answers across your documents.",
                                    className="section-description",
                                ),
                            ],
                            className="section-heading",
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id="semantic-search-query",
                                    type="text",
                                    placeholder="Try: What is the email address in my resume?",
                                    debounce=False,
                                    className="search-input",
                                ),
                                dcc.Input(
                                    id="semantic-search-topk",
                                    type="number",
                                    min=1,
                                    max=config.MAX_SEARCH_TOP_K,
                                    step=1,
                                    value=config.DEFAULT_SEARCH_TOP_K,
                                    className="top-k-input",
                                ),
                                html.Button(
                                    "Search files",
                                    id="semantic-search-button",
                                    n_clicks=0,
                                    className="button button-primary search-button",
                                ),
                            ],
                            className="search-controls",
                        ),
                        dcc.Loading(
                            type="circle",
                            children=[
                                html.Div(id="semantic-search-status", className="search-status"),
                                html.Div(id="semantic-search-answer", className="answer-region"),
                                html.Div(id="semantic-search-results", className="results-region"),
                            ],
                        ),
                    ],
                    className="surface-card search-card",
                ),
                dcc.Tabs(
                    value="map-tab",
                    parent_className="tabs-shell",
                    className="tabs",
                    content_className="tab-content",
                    colors={"border": "transparent", "primary": "#6d5ce8", "background": "#eef0fb"},
                    children=[
                        dcc.Tab(
                            label="Semantic Map",
                            value="map-tab",
                            className="tab",
                            selected_className="tab tab-selected",
                            children=[
                                html.Div(
                                    id="selected-file-label",
                                    className="selected-file-label",
                                    children="Select a file node to inspect conflict resolution options.",
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Open selected",
                                            id="open-selected-button",
                                            n_clicks=0,
                                            className="button button-secondary",
                                        ),
                                        html.Span(id="open-selected-status", className="inline-status"),
                                    ],
                                    className="file-actions",
                                ),
                                html.Div(
                                    children=[
                                        dcc.Input(
                                            id="rename-file-input",
                                            type="text",
                                            placeholder="New file name (e.g., notes_v2.txt)",
                                            className="rename-input",
                                        ),
                                        html.Button(
                                            "Rename Selected",
                                            id="rename-file-button",
                                            n_clicks=0,
                                            className="button button-secondary",
                                        ),
                                        html.Button(
                                            "Delete Selected",
                                            id="delete-file-button",
                                            n_clicks=0,
                                            className="button button-danger",
                                        ),
                                    ],
                                    className="file-actions",
                                ),
                                html.Div(id="file-operation-status", className="inline-status"),
                                html.Div(
                                    id="conflict-message",
                                    className="conflict-message",
                                ),
                                html.Div(
                                    children=[
                                        dcc.Dropdown(
                                            id="override-cluster-choice",
                                            options=[{"label": "Auto (No Override)", "value": "__auto__"}],
                                            value="__auto__",
                                            clearable=False,
                                            className="override-dropdown",
                                        ),
                                        html.Button(
                                            "Apply Override",
                                            id="apply-override-button",
                                            n_clicks=0,
                                            className="button button-secondary",
                                        ),
                                    ],
                                    className="override-controls",
                                ),
                                html.Div(id="override-status", className="inline-status"),
                                dcc.Graph(
                                    id="semantic-graph",
                                    className="chart-card",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                                html.Div(id="duplicates-panel", className="info-card"),
                                html.Div(id="summaries-panel", className="info-card"),
                            ],
                        ),
                        dcc.Tab(
                            label="Timeline",
                            value="timeline-tab",
                            className="tab",
                            selected_className="tab tab-selected",
                            children=[
                                dcc.Graph(
                                    id="timeline-graph",
                                    className="chart-card",
                                    config={"displayModeBar": True, "displaylogo": False},
                                ),
                                html.Div(id="events-panel", className="info-card"),
                            ],
                        ),
                    ],
                ),
                dcc.Interval(id="refresh-timer", interval=config.REFRESH_INTERVAL_MS, n_intervals=0),
            ],
            className="app-shell",
        )

    def _register_callbacks(self) -> None:
        @self.app.callback(
            Output("snapshot-store", "data"),
            Input("refresh-timer", "n_intervals"),
        )
        def refresh_snapshot(_: int) -> dict[str, Any]:
            return self.snapshot_provider()

        @self.app.callback(
            Output("semantic-graph", "figure"),
            Output("stats-panel", "children"),
            Input("snapshot-store", "data"),
        )
        def render_map_and_stats(snapshot: Any) -> tuple[go.Figure, Any]:
            safe_snapshot = self._coerce_snapshot(snapshot)
            figure = self._build_figure(safe_snapshot)
            stats = self._build_stats(safe_snapshot)
            return figure, stats

        @self.app.callback(
            Output("timeline-graph", "figure"),
            Input("snapshot-store", "data"),
        )
        def render_timeline(snapshot: Any) -> go.Figure:
            return self._build_timeline_figure(self._coerce_snapshot(snapshot))

        @self.app.callback(
            Output("duplicates-panel", "children"),
            Input("snapshot-store", "data"),
        )
        def render_duplicates(snapshot: Any) -> Any:
            return self._build_duplicates_panel(self._coerce_snapshot(snapshot))

        @self.app.callback(
            Output("summaries-panel", "children"),
            Input("snapshot-store", "data"),
        )
        def render_summaries(snapshot: Any) -> Any:
            return self._build_summaries_panel(self._coerce_snapshot(snapshot))

        @self.app.callback(
            Output("events-panel", "children"),
            Input("snapshot-store", "data"),
        )
        def render_events(snapshot: Any) -> Any:
            return self._build_events_panel(self._coerce_snapshot(snapshot))

        # Backward compatibility for stale browser tabs that still post the old
        # multi-output callback signature from earlier dashboard versions.
        @self.app.callback(
            Output("semantic-graph", "figure", allow_duplicate=True),
            Output("timeline-graph", "figure", allow_duplicate=True),
            Output("stats-panel", "children", allow_duplicate=True),
            Output("duplicates-panel", "children", allow_duplicate=True),
            Output("summaries-panel", "children", allow_duplicate=True),
            Output("events-panel", "children", allow_duplicate=True),
            Input("refresh-timer", "n_intervals"),
            prevent_initial_call=True,
        )
        def render_legacy_bundle(_tick: int) -> tuple[go.Figure, go.Figure, Any, Any, Any, Any]:
            snapshot = self._coerce_snapshot(None)
            return (
                self._build_figure(snapshot),
                self._build_timeline_figure(snapshot),
                self._build_stats(snapshot),
                self._build_duplicates_panel(snapshot),
                self._build_summaries_panel(snapshot),
                self._build_events_panel(snapshot),
            )

        self._install_legacy_callback_aliases()

        @self.app.callback(
            Output("selected-file-path", "data"),
            Input("semantic-graph", "clickData"),
            prevent_initial_call=True,
        )
        def select_file(click_data: dict[str, Any] | None) -> str:
            if not click_data:
                raise PreventUpdate
            points = click_data.get("points") or []
            if not points:
                raise PreventUpdate
            customdata = points[0].get("customdata")
            if not customdata or len(customdata) < 1:
                raise PreventUpdate
            return str(customdata[0])

        @self.app.callback(
            Output("selected-file-label", "children"),
            Output("conflict-message", "children"),
            Output("override-cluster-choice", "options"),
            Output("override-cluster-choice", "value"),
            Input("selected-file-path", "data"),
            Input("snapshot-store", "data"),
        )
        def update_conflict_ui(selected_path: str, snapshot: dict[str, Any]) -> tuple[str, str, list[dict[str, str]], str]:
            selected = str(selected_path or "")
            if not selected:
                options = [{"label": "Auto (No Override)", "value": "__auto__"}]
                return (
                    "Select a file node to inspect conflict resolution options.",
                    "",
                    options,
                    "__auto__",
                )

            points = snapshot.get("points", []) if snapshot else []
            point_lookup = {str(item.get("path", "")): item for item in points}
            point = point_lookup.get(selected, {})
            filename = str(point.get("filename", Path(selected).name))
            cluster = str(point.get("cluster", config.UNCATEGORIZED_DIR))
            label = f"Selected: {filename} | Current cluster: {cluster}"

            ambiguities = snapshot.get("ambiguities", {}) if snapshot else {}
            entry = ambiguities.get(selected, {})
            cluster_options = list(snapshot.get("cluster_options", [])) if snapshot else []

            options: list[dict[str, str]] = [{"label": "Auto (No Override)", "value": "__auto__"}]
            for name in cluster_options:
                options.append({"label": f"Override -> {name}", "value": str(name)})

            conflict_message = ""
            value = "__auto__"
            if entry:
                choices = [str(item) for item in entry.get("choices", []) if str(item).strip()]
                if choices:
                    conflict_message = (
                        f"{filename} could belong to {choices[0]} or {choices[1] if len(choices) > 1 else choices[0]}. "
                        "Choose your preference."
                    )
                    options = [{"label": "Auto (No Override)", "value": "__auto__"}] + [
                        {"label": f"Prefer {choice}", "value": choice}
                        for choice in choices
                    ] + [
                        {"label": f"Override -> {name}", "value": str(name)}
                        for name in cluster_options
                        if name not in choices
                    ]

            if bool(point.get("manual_override")):
                value = cluster

            return label, conflict_message, options, value

        @self.app.callback(
            Output("override-status", "children"),
            Input("apply-override-button", "n_clicks"),
            State("selected-file-path", "data"),
            State("override-cluster-choice", "value"),
            prevent_initial_call=True,
        )
        def apply_override(_clicks: int, selected_path: str, chosen_cluster: str | None) -> str:
            if not selected_path:
                return "Select a file before applying override."
            if self.override_handler is None:
                return "Override handler is not configured."

            choice = str(chosen_cluster or "__auto__")
            target = None if choice == "__auto__" else choice

            try:
                ok, message = self.override_handler(selected_path, target)
            except Exception as exc:
                self.logger.exception("Failed to apply manual override")
                return f"Override failed: {exc}"
            return message if ok else f"Override failed: {message}"

        @self.app.callback(
            Output("open-selected-status", "children"),
            Input("open-selected-button", "n_clicks"),
            State("selected-file-path", "data"),
            prevent_initial_call=True,
        )
        def open_selected_file(_clicks: int, selected_path: str) -> str:
            if not selected_path:
                return "Select a file first."
            success, message = self._open_file(str(selected_path))
            if success:
                return f"Opened: {selected_path}"
            return f"Open failed: {message}"

        @self.app.callback(
            Output("file-operation-status", "children"),
            Output("selected-file-path", "data", allow_duplicate=True),
            Output("snapshot-store", "data", allow_duplicate=True),
            Input("rename-file-button", "n_clicks"),
            State("selected-file-path", "data"),
            State("rename-file-input", "value"),
            prevent_initial_call=True,
        )
        def rename_selected_file(
            _clicks: int,
            selected_path: str,
            new_name: str | None,
        ) -> tuple[str, str | Any, dict[str, Any] | Any]:
            if not selected_path:
                return "Select a file first.", no_update, no_update

            success, message, new_path = self._rename_file(selected_path, str(new_name or ""))
            if not success:
                return f"Rename failed: {message}", no_update, no_update

            if self.rename_handler is not None:
                try:
                    self.rename_handler(selected_path, new_path)
                except Exception as exc:
                    self.logger.exception("Rename handler failed")
                    return f"Renamed, but semantic update failed: {exc}", new_path, self._coerce_snapshot(None)

            return message, new_path, self._coerce_snapshot(None)

        @self.app.callback(
            Output("file-operation-status", "children", allow_duplicate=True),
            Output("selected-file-path", "data", allow_duplicate=True),
            Output("snapshot-store", "data", allow_duplicate=True),
            Input("delete-file-button", "n_clicks"),
            State("selected-file-path", "data"),
            prevent_initial_call=True,
        )
        def delete_selected_file(
            _clicks: int,
            selected_path: str,
        ) -> tuple[str, str, dict[str, Any] | Any]:
            if not selected_path:
                return "Select a file first.", "", no_update

            success, message = self._delete_file(selected_path)
            if not success:
                return f"Delete failed: {message}", selected_path, no_update

            if self.delete_handler is not None:
                try:
                    self.delete_handler(selected_path)
                except Exception as exc:
                    self.logger.exception("Delete handler failed")
                    return f"Deleted, but semantic update failed: {exc}", "", self._coerce_snapshot(None)

            return message, "", self._coerce_snapshot(None)

        @self.app.callback(
            Output("semantic-search-status", "children"),
            Output("semantic-search-answer", "children"),
            Output("semantic-search-results", "children"),
            Input("semantic-search-button", "n_clicks"),
            Input("semantic-search-query", "n_submit"),
            State("semantic-search-query", "value"),
            State("semantic-search-topk", "value"),
            prevent_initial_call=True,
        )
        def run_semantic_search(
            _clicks: int,
            _submitted: int,
            query: str | None,
            top_k: int | None,
        ) -> tuple[str, Any, Any]:
            normalized_query = str(query or "").strip()
            if not normalized_query:
                return "Enter a query first.", "", ""

            if self.query_handler is None:
                return "Search handler is not configured.", "", ""

            safe_top_k = int(top_k) if top_k else config.DEFAULT_SEARCH_TOP_K
            safe_top_k = max(1, min(safe_top_k, config.MAX_SEARCH_TOP_K))

            try:
                response = self.query_handler(normalized_query, safe_top_k)
            except Exception as exc:
                self.logger.exception("Search query failed")
                return f"Search for '{normalized_query}' failed: {exc}", "", ""

            results = response.get("results", [])
            answer = str(response.get("answer", "")).strip()
            error = response.get("error")
            generator = str(response.get("generator", "none"))

            if error:
                status = f"Results for '{normalized_query}': retrieved {len(results)}. Generation error: {error}"
            else:
                provider = generator if generator != "none" else "disabled"
                status = f"Results for '{normalized_query}': retrieved {len(results)}. Generator: {provider}"

            answer_block: Any
            if answer:
                answer_block = html.Div(
                    children=[
                        html.Div("ANSWER", className="section-eyebrow"),
                        html.Pre(
                            answer,
                            className="answer-text",
                        ),
                    ]
                )
            else:
                answer_block = ""

            results_block = self._build_search_results(results)
            return status, answer_block, results_block

    def _build_figure(self, snapshot: dict[str, Any]) -> go.Figure:
        figure = go.Figure()
        points = snapshot.get("points", [])
        edges = snapshot.get("similarity_edges", [])
        summaries = snapshot.get("cluster_summaries", {})

        if not points:
            message = str(snapshot.get("empty_message", "No data available."))
            figure.add_annotation(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 16},
            )
            figure.update_layout(
                title="Semantic Map",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                template="plotly_white",
            )
            return figure

        coordinate_lookup = {
            str(point.get("path", "")): (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
            for point in points
        }

        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in coordinate_lookup or target not in coordinate_lookup:
                continue

            x1, y1 = coordinate_lookup[source]
            x2, y2 = coordinate_lookup[target]
            similarity = float(edge.get("similarity", 0.0))
            width = 0.6 + (2.6 * max(0.0, min(1.0, similarity)))
            figure.add_trace(
                go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    line={"color": "rgba(120,120,120,0.30)", "width": width},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        clusters = sorted({str(point.get("cluster", config.UNCATEGORIZED_DIR)) for point in points})
        palette = [
            "#6655d8",
            "#ef8b45",
            "#32a98f",
            "#e56d8b",
            "#438ed0",
            "#a36ac7",
            "#d0a33f",
            "#4eaaa9",
            "#d46d4a",
            "#7480cb",
        ]

        for index, cluster in enumerate(clusters):
            cluster_points = [point for point in points if point.get("cluster") == cluster]
            x_values = [float(point.get("x", 0.0)) for point in cluster_points]
            y_values = [float(point.get("y", 0.0)) for point in cluster_points]
            filenames = [str(point.get("filename", "")) for point in cluster_points]

            marker_colors = [
                "#f7d154" if bool(point.get("ambiguous")) else palette[index % len(palette)]
                for point in cluster_points
            ]
            marker_sizes = [14 if bool(point.get("ambiguous")) else 10 for point in cluster_points]
            marker_symbols = ["diamond" if bool(point.get("ambiguous")) else "circle" for point in cluster_points]

            customdata = [
                [
                    str(point.get("path", "")),
                    cluster,
                    self._format_size(int(point.get("size", 0))),
                    str(point.get("mtime_iso", "")),
                    self._trim_text(str(point.get("snippet", "")), max_chars=120),
                    self._trim_text(str(summaries.get(cluster, "")), max_chars=120),
                    "yes" if bool(point.get("ambiguous")) else "no",
                ]
                for point in cluster_points
            ]

            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name=cluster,
                    text=filenames,
                    customdata=customdata,
                    marker={
                        "size": marker_sizes,
                        "opacity": 0.85,
                        "color": marker_colors,
                        "symbol": marker_symbols,
                        "line": {"width": 2, "color": "#ffffff"},
                    },
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Cluster: %{customdata[1]}<br>"
                        "Ambiguous: %{customdata[6]}<br>"
                        "Size: %{customdata[2]}<br>"
                        "Snippet: %{customdata[4]}<br>"
                        "Cluster summary: %{customdata[5]}"
                        "<extra></extra>"
                    ),
                )
            )

        figure.update_layout(
            title="Semantic Map (Yellow = Ambiguous Candidate)",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            template="plotly_white",
            legend_title="Clusters",
            hoverlabel={"align": "left"},
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "color": "#26324b"},
            margin={"l": 48, "r": 24, "t": 64, "b": 48},
        )
        return figure

    def _build_timeline_figure(self, snapshot: dict[str, Any]) -> go.Figure:
        figure = go.Figure()
        timeline = snapshot.get("timeline", [])
        if not timeline:
            figure.add_annotation(
                text="No timeline points yet.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
            figure.update_layout(
                title="Timeline",
                xaxis_title="Time",
                yaxis_title="Count",
                template="plotly_white",
            )
            return figure

        times: list[datetime] = []
        total_files: list[int] = []
        cluster_counts: list[int] = []
        for entry in timeline:
            try:
                parsed = datetime.fromisoformat(str(entry.get("time_iso", "")))
            except ValueError:
                continue
            times.append(parsed)
            total_files.append(int(entry.get("total_files", 0)))
            cluster_counts.append(int(entry.get("cluster_count", 0)))

        if not times:
            return figure

        figure.add_trace(
            go.Scatter(
                x=times,
                y=total_files,
                mode="lines+markers",
                name="Total files",
                line={"width": 3, "color": "#6d5ce8", "shape": "spline"},
                marker={"size": 8, "color": "#6d5ce8", "line": {"width": 2, "color": "#ffffff"}},
                fill="tozeroy",
                fillcolor="rgba(109,92,232,0.08)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=cluster_counts,
                mode="lines+markers",
                name="Semantic clusters",
                line={"width": 3, "color": "#38a995", "shape": "spline"},
                marker={"size": 8, "color": "#38a995", "line": {"width": 2, "color": "#ffffff"}},
            )
        )
        figure.update_layout(
            title="Temporal View: File/Cluster Evolution",
            xaxis_title="Time",
            yaxis_title="Count",
            template="plotly_white",
            legend_title="Metric",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "color": "#26324b"},
            margin={"l": 48, "r": 24, "t": 64, "b": 48},
        )
        return figure

    def _build_stats(self, snapshot: dict[str, Any]) -> html.Div:
        stats = snapshot.get("stats", {})
        last_update = str(stats.get("last_update_iso", ""))
        if last_update:
            try:
                parsed = datetime.fromisoformat(last_update)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                last_update = parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            except ValueError:
                pass

        ambiguous_count = sum(1 for point in snapshot.get("points", []) if bool(point.get("ambiguous")))
        duplicate_count = len(snapshot.get("duplicates", []))

        metrics = [
            ("Indexed files", int(stats.get("total_files", 0)), "metric-violet"),
            ("Semantic clusters", int(stats.get("cluster_count", 0)), "metric-blue"),
            ("Uncategorized", int(stats.get("uncategorized_count", 0)), "metric-amber"),
            ("Ambiguous files", ambiguous_count, "metric-rose"),
            ("Possible duplicates", duplicate_count, "metric-teal"),
        ]
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(str(value), className="metric-value"),
                        html.Div(label, className="metric-label"),
                    ],
                    className=f"metric-card {color_class}",
                )
                for label, value, color_class in metrics
            ]
            + [
                html.Div(
                    [
                        html.Div("LIVE INDEX", className="metric-live-label"),
                        html.Div(last_update or "Waiting for first update", className="metric-live-value"),
                    ],
                    className="metric-card metric-live",
                )
            ],
            className="stats-grid",
        )

    def _build_duplicates_panel(self, snapshot: dict[str, Any]) -> Any:
        duplicates = snapshot.get("duplicates", [])
        if not duplicates:
            return html.Div("No near-duplicates detected above threshold.")

        rows: list[Any] = []
        for item in duplicates[:10]:
            rows.append(
                html.Li(
                    f"{item.get('filename_a')} <-> {item.get('filename_b')} "
                    f"| similarity={float(item.get('similarity', 0.0)):.3f} "
                    f"| {item.get('cluster_a')} / {item.get('cluster_b')}",
                    className="info-list-item",
                )
            )
        return html.Div(
            [
                html.Div("REVIEW", className="section-eyebrow"),
                html.H3("Near-duplicate alerts", className="info-title"),
                html.P(
                    "Pairs above threshold may be duplicates. Review them before removing either file.",
                    className="info-description",
                ),
                html.Ul(rows, className="info-list"),
            ],
            className="info-card-content",
        )

    def _build_summaries_panel(self, snapshot: dict[str, Any]) -> Any:
        summaries = snapshot.get("cluster_summaries", {})
        if not summaries:
            return html.Div("Cluster summaries unavailable.", className="empty-note")

        rows = [
            html.Li(
                [
                    html.Strong(cluster, className="summary-cluster"),
                    html.Span(summary),
                ],
                className="info-list-item summary-item",
            )
            for cluster, summary in sorted(summaries.items(), key=lambda item: item[0].lower())
        ]
        return html.Div(
            [
                html.Div("DISCOVER", className="section-eyebrow"),
                html.H3("Cluster summaries", className="info-title"),
                html.Ul(rows, className="info-list"),
            ],
            className="info-card-content",
        )

    def _build_events_panel(self, snapshot: dict[str, Any]) -> Any:
        events = snapshot.get("recent_events", [])
        if not events:
            return html.Div("No semantic events yet.", className="empty-note")

        rows: list[Any] = []
        for event in reversed(events):
            rows.append(
                html.Li(
                    f"[{event.get('type')}] {event.get('detail')} "
                    f"(drift={float(event.get('semantic_drift', 0.0)):.2f})",
                    className="info-list-item",
                )
            )
        return html.Div(
            [
                html.Div("ACTIVITY", className="section-eyebrow"),
                html.H3("Recent semantic events", className="info-title"),
                html.Ul(rows, className="info-list"),
            ],
            className="info-card-content",
        )

    def _build_search_results(self, results: list[dict[str, Any]]) -> Any:
        if not results:
            return html.Div("No search results yet.", className="empty-note")

        rows: list[Any] = []
        for index, item in enumerate(results, start=1):
            rows.append(
                html.Li(
                    [
                        html.Div(
                            [
                                html.Span(f"{index:02}", className="result-number"),
                                html.Div(
                                    [
                                        html.Strong(
                                            str(item.get("filename", "unknown")),
                                            className="result-filename",
                                        ),
                                        html.Span(
                                            f"{item.get('cluster', config.UNCATEGORIZED_DIR)}"
                                            f"  /  match {float(item.get('relevance', 0.0)):.0%}",
                                            className="result-meta",
                                        ),
                                    ],
                                    className="result-heading",
                                ),
                            ],
                            className="result-title-row",
                        ),
                        html.Div(str(item.get("snippet", "")), className="result-snippet"),
                        html.Code(str(item.get("path", "")), className="result-path"),
                    ],
                    className="result-card",
                )
            )
        return html.Div(
            [
                html.Div("MATCHING DOCUMENTS", className="section-eyebrow"),
                html.H3("Retrieved context", className="info-title"),
                html.Ol(rows, className="results-list"),
            ],
            className="info-card-content",
        )

    def _trim_text(self, text: str, max_chars: int) -> str:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[: max_chars - 3]}..."

    def _install_legacy_callback_aliases(self) -> None:
        """Alias historical callback keys so stale tabs continue to function."""
        legacy_key = (
            "..semantic-graph.figure...timeline-graph.figure...stats-panel.children..."
            "duplicates-panel.children...summaries-panel.children...events-panel.children.."
        )
        if legacy_key in self.app.callback_map:
            return

        for key, value in list(self.app.callback_map.items()):
            if (
                key.startswith("..semantic-graph.figure@")
                and "...timeline-graph.figure@" in key
                and "...stats-panel.children@" in key
                and "...duplicates-panel.children@" in key
                and "...summaries-panel.children@" in key
                and "...events-panel.children@" in key
            ):
                self.app.callback_map[legacy_key] = value
                break

    def _coerce_snapshot(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        try:
            snapshot = self.snapshot_provider()
            if isinstance(snapshot, dict):
                return snapshot
        except Exception:
            self.logger.exception("Failed to fetch live snapshot; using empty snapshot fallback")
        return {}

    def _rename_file(self, path: str, new_name: str) -> tuple[bool, str, str]:
        original = Path(path)
        if not original.exists() or not original.is_file():
            return False, "Selected file no longer exists.", path
        if config.RAW_SUBDIR not in original.parts:
            return False, "Only files inside _raw can be renamed.", path

        candidate = new_name.strip()
        if not candidate:
            return False, "Enter a new file name.", path
        if "/" in candidate or "\\" in candidate:
            return False, "New name must not contain path separators.", path

        if candidate in {".", ".."}:
            return False, "Invalid file name.", path

        if "." not in Path(candidate).name and original.suffix:
            candidate = f"{candidate}{original.suffix}"

        target = original.with_name(candidate)
        if target.exists():
            return False, "A file with that name already exists.", path

        try:
            original.rename(target)
            return True, f"Renamed to: {target.name}", str(target.resolve(strict=False))
        except Exception as exc:
            self.logger.warning("Failed to rename file %s -> %s: %s", original, target, exc)
            return False, str(exc), path

    def _delete_file(self, path: str) -> tuple[bool, str]:
        target = Path(path)
        if not target.exists():
            return False, "Selected file no longer exists."
        if not target.is_file():
            return False, "Selected path is not a file."
        if config.RAW_SUBDIR not in target.parts:
            return False, "Only files inside _raw can be deleted."

        try:
            target.unlink()
            return True, f"Deleted: {target.name}"
        except Exception as exc:
            self.logger.warning("Failed to delete file %s: %s", target, exc)
            return False, str(exc)

    def _open_file(self, path: str) -> tuple[bool, str]:
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif os.name == "posix":
                command = ["open", path] if self._is_macos() else ["xdg-open", path]
                subprocess.Popen(command)
            else:
                return False, f"Unsupported platform: {os.name}"
            return True, ""
        except Exception as exc:
            self.logger.warning("Failed to open file %s: %s", path, exc)
            return False, str(exc)

    def _is_macos(self) -> bool:
        return os.uname().sysname.lower() == "darwin" if hasattr(os, "uname") else False

    def _format_size(self, size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"
