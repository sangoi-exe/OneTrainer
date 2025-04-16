import json
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import re
from collections import defaultdict # Importar defaultdict para simplificar a agregação

ARQUIVO_JSON = "F:/OT/training_deltas/Training_Deltas_20250414_160208.json"
TIPO_COR_PADRAO = "#1f77b4"

with open(ARQUIVO_JSON) as f:
    raw_data = json.load(f)

epochs = sorted(raw_data.keys(), key=lambda x: int(x.split("_")[1]))

all_layers = set()
for epoch in epochs:
    all_layers.update(raw_data[epoch].keys())

# Usa defaultdict para facilitar a construção das listas de valores por camada
layer_data_dd = defaultdict(list)

# Itera pelas épocas ordenadas e preenche os dados para todas as camadas
for epoch in epochs:
    epoch_data = raw_data[epoch]
    for layer in all_layers:
        # Adiciona o valor da camada para a época atual, ou 0.0 se não existir
        layer_data_dd[layer].append(epoch_data.get(layer, 0.0))

# Converte defaultdict para dict padrão para o resto do código
layer_data = dict(layer_data_dd)
layers = all_layers # Usar o conjunto completo de camadas identificadas


# === FUNÇÕES AUXILIARES ===
def extrair_tipo(nome: str):
    m = re.search(r"(attn1|attn2|ff_net|proj|to_(?:q|k|v|out)|resnets|conv\d+)", nome)
    return m.group(1) if m else "outros"


tipos_map = defaultdict(list)
for layer in layers: # Usa o conjunto 'layers' que agora contém todas as camadas únicas
    tipo = extrair_tipo(layer)
    tipos_map[tipo].append(layer)

# === DASH APP ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Visualizador de Deltas por Camada"

app.layout = dbc.Container(
    [
        html.H3("Análise de Deltas por Módulo", className="my-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Tipo de módulo"),
                        dcc.Dropdown(
                            id="tipo-modulo",
                            options=[
                                {"label": t, "value": t} for t in sorted(tipos_map)
                            ],
                            value=None,
                            placeholder="Selecione o tipo (ex: attn1)",
                            clearable=True,
                            searchable=True,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Camada / Módulo"),
                        dcc.Dropdown(
                            id="layer-dropdown",
                            options=[],
                            placeholder="Selecione a camada",
                            clearable=True,
                            searchable=True,
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Label("Cor da linha"),
                        dbc.Input(id="input-cor", type="color", value=TIPO_COR_PADRAO),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Br(),
                        dbc.Button(
                            "Adicionar ao gráfico",
                            id="btn-adicionar",
                            color="primary",
                            className="w-100",
                        ),
                    ],
                    width=1,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Intervalo de épocas"),
                        dcc.RangeSlider(
                            id="slider-epocas",
                            min=0,
                            max=len(epochs) - 1,
                            step=1,
                            marks={
                                i: str(i)
                                for i in range(
                                    0, len(epochs), max(1, len(epochs) // 10)
                                )
                            },
                            value=[0, len(epochs) - 1],
                        ),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Inserir todos",
                            id="btn-todos",
                            color="success",
                            className="me-2",
                        ),
                        dbc.Button("Remover todos", id="btn-limpar", color="danger"),
                        html.Div(id="linha-selecionada", className="mt-2 text-info"),
                    ]
                )
            ],
            className="my-3",
        ),
        dcc.Graph(id="grafico", config={"displayModeBar": False}),
        dcc.Store(id="linhas-grafico", data=[]),
    ],
    fluid=True,
)

# === CALLBACKS ===


@app.callback(Output("layer-dropdown", "options"), Input("tipo-modulo", "value"))
def atualizar_layers(tipo):
    if tipo is None:
        return []
    return [{"label": l, "value": l} for l in sorted(tipos_map.get(tipo, []))]


@app.callback(
    Output("linhas-grafico", "data"),
    Input("btn-adicionar", "n_clicks"),
    State("layer-dropdown", "value"),
    State("input-cor", "value"),
    State("linhas-grafico", "data"),
    prevent_initial_call=True,
)
def adicionar_linha(_, layer, cor, linhas):
    if layer and layer not in [l["name"] for l in linhas]:
        linhas.append({"name": layer, "color": cor})
    return linhas


@app.callback(
    Output("linhas-grafico", "data", allow_duplicate=True),
    Input("btn-limpar", "n_clicks"),
    prevent_initial_call=True,
)
def limpar_linhas(_):
    return []


@app.callback(
    Output("linhas-grafico", "data", allow_duplicate=True),
    Input("btn-todos", "n_clicks"),
    State("tipo-modulo", "value"),
    prevent_initial_call=True,
)
def adicionar_todos(_, tipo):
    if tipo is None:
        return []
    return [{"name": l, "color": TIPO_COR_PADRAO} for l in tipos_map[tipo]]


@app.callback(
    Output("grafico", "figure"),
    Input("linhas-grafico", "data"),
    Input("slider-epocas", "value"),
)
def atualizar_grafico(linhas, intervalo):
    fig = go.Figure()
    x = list(range(intervalo[0], intervalo[1] + 1))
    for linha in linhas:
        y = layer_data[linha["name"]][intervalo[0] : intervalo[1] + 1]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=linha["name"],
                line=dict(color=linha["color"]),
            )
        )

    fig.update_layout(template="plotly_dark", height=600, clickmode="event+select")
    fig.update_xaxes(title="Época")
    fig.update_yaxes(title="Norma L2 do Delta")
    return fig


@app.callback(
    Output("linha-selecionada", "children"),
    Output("layer-dropdown", "value"),
    Input("grafico", "clickData"),
    prevent_initial_call=True,
)
def selecionar_linha(clickData):
    if clickData:
        nome = clickData["points"][0]["curveNumber"]
        texto = clickData["points"][0]["data"]["name"]
        return f"Linha selecionada: {texto}", texto
    return "", None


if __name__ == "__main__":
    app.run(debug=True)
