import numpy as np
import pandas as pd
import pytest
from unittest import mock

from plotly.graph_objects import Scatter
from plotly.graph_objects import Contour
from neat_ml import figure_plotting as fp

@pytest.fixture(scope="session")
def sample_df():
    """Small synthetic composition/phase DataFrame (single-phase=0, two-phase=1)."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 20, 20)
    y = rng.uniform(0, 20, 20)
    phase = (x + y > 20).astype(int)          
    return pd.DataFrame(
        {"x": x, "y": y, "phase": phase}
    )

@pytest.fixture
def no_gui():
    """Patch go.Figure.show so the test run stays headless."""
    with mock.patch("plotly.graph_objects.Figure.show"):
        yield

def test_read_composition_excel_creates_csv(tmp_path, sample_df):
    xlsx_path = tmp_path / "toy.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(sample_df[["x", "y", "phase"]]).to_excel(
            writer, sheet_name="SheetA", index=False, header=False, startrow=1
        )
    out_csv = tmp_path / "out.csv"
    fp.read_composition_excel(
        file_path=str(xlsx_path),
        excel_sheet_name="SheetA",
        xcolum="Xval",
        ycolumn="Yval",
        phase_column="PhaseVal",
        output_path=str(out_csv),
    )
    assert out_csv.exists(), "CSV was not created"
    df_out = pd.read_csv(out_csv)
    assert list(df_out.columns[1:]) == ["Xval", "Yval", "PhaseVal"]

@pytest.mark.parametrize("plot_regions", [True, False])
def test_plot_gmm_decision_regions_returns_valid_objects(sample_df, plot_regions):
    gmm, labels, boundary_pts, traces = fp.plot_gmm_decision_regions(
        df=sample_df.rename(columns={"x": "X", "y": "Y", "phase": "Phase"}),
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        xrange=[0, 20],
        yrange=[0, 20],
        n_components=2,
        random_state=42,
        plot_regions=plot_regions,
    )
    assert labels.shape[0] == len(sample_df)
    if plot_regions:
        assert isinstance(traces[0], Contour)
    if boundary_pts is not None:
        assert boundary_pts.shape[1] == 2

def test_plot_gmm_composition_phase_scatter_trace(sample_df):
    gmm, feats, boundary, traces = fp.plot_gmm_composition_phase(
        df=sample_df.rename(columns={"x": "X", "y": "Y", "phase": "Phase"}),
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        xrange=[0, 20],
        yrange=[0, 20],
        point_cmap=["blue", "lightgrey"],
        plot_regions=False,
    )
    assert all(isinstance(t, Scatter) for t in traces)
    assert feats.shape == (len(sample_df),)

@pytest.mark.parametrize(
    "wrapper,kwargs",
    [
        (fp.titration_diagram,
         dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 10], yrange=[0, 10])),
        (fp.mathematical_model,
         dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 10], yrange=[0, 10])),
    ],
)
def test_diagram_writers_create_png(tmp_path, sample_df, no_gui, wrapper, kwargs):
    csv_path = tmp_path / "toy.csv"
    sample_df.rename(columns={"x": "X", "y": "Y", "phase": "Phase"}).to_csv(csv_path, index=False)
    out_png = tmp_path / "out.png"

    wrapper(
        file_path=str(csv_path),
        output_path=str(out_png),
        **kwargs,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0     

def test_plot_two_scatter_creates_png(tmp_path, no_gui):
    xls_path = tmp_path / "toy.xlsx"
    with pd.ExcelWriter(xls_path, engine="openpyxl") as writer:
        df1 = pd.DataFrame({"X": [1, 2, 3], "Y": [3, 2, 1]})
        df2 = pd.DataFrame({"X": [1, 2, 3], "Y": [1, 2, 3]})
        df1.to_excel(writer, sheet_name="sheet_one", index=False)
        df2.to_excel(writer, sheet_name="sheet_two", index=False)

    out_png = tmp_path / "scatter.png"
    fp.plot_two_scatter(
        excel_file=str(xls_path),
        sheet1="sheet_one",
        sheet2="sheet_two",
        output_path=str(out_png),
        xlim=[0, 4],
        ylim=[0, 4],
    )
    assert out_png.exists()
    assert out_png.stat().st_size > 0

def test_phase_diagram_exp_creates_png(tmp_path, sample_df, no_gui):
    """
    Smoke-test the high-level phase_diagram_exp() helper:
    it should run without error and create a non-empty image file.
    """
    csv_path = tmp_path / "phase_input.csv"
    sample_df.rename(columns={"x": "X", "y": "Y", "phase": "Phase"}).to_csv(
        csv_path, index=False
    )
    out_png = tmp_path / "phase_diagram_exp.png"

    fp.phase_diagram_exp(
        file_path=str(csv_path),
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        xrange=[0, 20],
        yrange=[0, 20],
        output_path=str(out_png),
    )

    assert out_png.exists(), "PNG was not written"
    assert out_png.stat().st_size > 0, "PNG is empty"
