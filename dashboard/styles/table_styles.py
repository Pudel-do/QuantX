# table_styles.py
import pandas as pd
from misc.utils import read_json


class PerformanceTableStyler:

    def __init__(self):
        self.const_cols = read_json("constant.json")["dashboard"]

    def build_styles(self, rows):
        if not rows:
            return []

        df = pd.DataFrame(rows)
        styles = []

        for col in [
            self.const_cols["ann_mean_ret"], 
            self.const_cols["total_ret"]
            ]:
            styles += self._diverging(df, col)

        styles += self._heatbar(df, self.const_cols["ann_vola"])

        return styles

    def _diverging(self, df, col):
        max_abs = df[col].abs().max()
        styles = []

        for i, val in enumerate(df[col]):
            color = "green" if val > 0 else "red"
            opacity = abs(val) / max_abs if max_abs else 0

            styles.append({
                "if": {"row_index": i, "column_id": col},
                "color": "black",
                "backgroundColor": f"rgba(0, 128, 0, {opacity})"
                if val > 0 else f"rgba(200, 0, 0, {opacity})"
            })

        return styles

    def _heatbar(self, df, col):
        min_val = df[col].min()
        max_val = df[col].max()
        styles = []

        for i, val in enumerate(df[col]):

            norm = (val - min_val) / (max_val - min_val) if max_val != min_val else 0

            intensity = 1 - norm

            styles.append({
                "if": {"row_index": i, "column_id": col},
                "backgroundColor": f"rgba(0, 150, 0, {0.15 + 0.75 * intensity})",
                "color": "black"
            })

        return styles
