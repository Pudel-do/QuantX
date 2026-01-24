

def build_plotly_annotation(start, end):
    return dict(
        text=f"<b>Zeitraum</b><br>{start} – {end}",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(
            size=14,
            color="black"
        ),
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=0
    )