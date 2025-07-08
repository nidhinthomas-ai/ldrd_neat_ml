import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Contour, Scatter
from skimage import measure
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

from typing import List, Tuple, Optional, Union

import plotly.io as pio
pio.renderers.default = "browser"

def read_composition_excel(
    file_path: str,
    excel_sheet_name: str,
    xcolum: str,
    ycolumn: str,
    phase_column: str,
    output_path: str
) -> pd.DataFrame:
    """
    Read composition data from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    excel_sheet_name : str
        Name of the sheet to read.
    xcolum : str
        Column label for x values.
    ycolumn : str
        Column label for y values.
    phase_column : str
        Column label for phase data.
    output_path : str
        Path to the .csv file

    Returns
    -------
        None
    """
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=excel_sheet_name,
            header=None,
            skiprows=1,
        )
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.",
              file=sys.stderr)
        sys.exit(1)

    result_df = pd.DataFrame({
        xcolum: df[0],
        ycolumn: df[1],
        phase_column: df[2],
    })

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    result_df.to_csv(output_path)


class GMMWrapper:
    """
    A wrapper to use a GMM trained on feature data as a 
    classifier in composition space.

    This class bridges the gap between a GMM trained on 
    1D phase data and the 2D composition space of a phase 
    iagram. For any given composition point (x, y), it 
    finds the nearest experimental data point in the 
    composition space and uses that point's phase to predict
    a cluster label with the GMM.

    Attributes:
        gmm (GaussianMixture): The pre-trained Gaussian 
                               Mixture Model.
        x_comp (np.ndarray): The 2D array of composition
                             data (n_samples, 2).
        x_features (np.ndarray): The corresponding 
                                 feature vectors used for 
                                 GMM training.
    """

    def __init__(
        self,
        gmm: GaussianMixture,
        x_comp: np.ndarray,
        x_features: np.ndarray
    ):
        """
        Initializes the GMMWrapper.

        Parameters:
        ----------
            gmm (GaussianMixture): The trained GMM instance.
            x_comp (np.ndarray): The composition data array 
                                 (n_samples, 2).
            x_features (np.ndarray): The feature data array 
                                     (n_samples, 1).
        """
        self.gmm = gmm
        self.x_comp = x_comp
        self.x_features = x_features

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predicts cluster labels for new composition points.

        Parameters:
        -----------
            x_test (np.ndarray): An array of new composition
                                 points (n_test_samples, 2).

        Returns:
        -------
            np.ndarray: The predicted GMM cluster labels 
                        for the input points.
        """
        closest_indices, _ = pairwise_distances_argmin_min(
            x_test, 
            self.x_comp
            )
        feature_vectors = self.x_features[closest_indices]
        labels = self.gmm.predict(feature_vectors)
        return labels


def extract_boundary_from_contour(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    level: float = 0.5
) -> Optional[np.ndarray]:
    """
    Extracts the longest boundary contour from a grid at 
    a specified level.

    Parameters:
    ----------
        z (np.ndarray): The 2D grid of predicted values.
        xs (np.ndarray): The x-coordinates corresponding
                         to the grid columns.
        ys (np.ndarray): The y-coordinates corresponding 
                         to the grid rows.
        level (float): The contour level to extract.

    Returns:
    --------
        Optional[np.ndarray]: An array of (x, y) coordinates
                              for the boundary, or None if 
                              no contour is found.
    """
    contours = measure.find_contours(z, level)
    if not contours:
        return None
    longest_contour = max(contours, key=len)
    resolution_x, resolution_y = len(xs), len(ys)
    boundary_points = np.column_stack((
        np.interp(longest_contour[:, 1], 
                  np.arange(resolution_x), xs),
        np.interp(longest_contour[:, 0], 
                  np.arange(resolution_y), ys)
    ))
    return boundary_points

def _standardise_labels(
    cluster_labels: np.ndarray, x_comp: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Remap raw GMM labels so conventions never flip.

    *Standard convention used downstream*

    ---------  ----------------------------
    label=0    Two Phase   (triangle-up, turquoise)
    label=1    Single Phase (square, light-steel-blue)
    ---------  ----------------------------

    Parameters
    ----------
    cluster_labels :
        Original labels from :pyclass:`sklearn.mixture.GaussianMixture`.
    x_comp :
        Composition coordinates associated with each label.

    Returns
    -------
    std_labels :
        Relabelled array following the convention above.
    label_map :
        Dict mapping raw → standard labels; apply to any
        future predictions (`z` grids, etc.).
    """
    centroids = {
        lbl: x_comp[cluster_labels == lbl].mean(axis=0)
        for lbl in np.unique(cluster_labels)
    }
    distances = {lbl: np.linalg.norm(c) for lbl, c in centroids.items()}

    near_lbl = min(distances, key=distances.get)
    far_lbl = max(distances, key=distances.get)

    label_map = {far_lbl: 0, near_lbl: 1}
    std_labels = np.vectorize(label_map.get)(cluster_labels)
    return std_labels, label_map

def plot_gmm_decision_regions(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange=List[int],
    yrange=List[int],
    n_components: int = 2,
    random_state: int = 42,
    region_colors: Optional[List[str]] = None,
    boundary_color: str = "red",
    resolution: int = 200,
    decision_alpha: float = 0.3,
    plot_regions: bool = True
) -> Tuple[GaussianMixture, np.ndarray, Optional[np.ndarray], List[Contour]]:
    """
    Trains a GMM and creates contour traces for phase 
    regions and boundaries.

    Parameters:
    -----------
        df (pd.DataFrame): DataFrame with composition and
                           phase data.
        x_col (str): Name of the column for the x-axis 
                     component.
        y_col (str): Name of the column for the y-axis 
                     component.
        phase_col (str): Name of the column containing 
                         phase information.
        xrange (List[int]): X-axis composition range.
        yrange (List[int]): Y-axis composition range. 
        n_components (int): Number of GMM components.
        random_state (int): Seed for reproducibility.
        region_colors (Optional[List[str]]): Colors for the 
                                             phase regions.
        boundary_color (str): Color for the boundary line.
        resolution (int): Grid resolution for the contour plot.
        decision_alpha (float): Opacity of the filled regions.
        plot_regions (bool): Whether to plot the filled regions.

    Returns:
    --------
        Tuple: Containing the GMM model, cluster labels, 
               boundary points, and a list of Plotly contour
               traces.
    """
    df_local = df.copy()
    x_features = df_local[phase_col].to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state
    )
    raw_labels = gmm.fit_predict(x_features)
    x_comp = df_local[[x_col, y_col]].to_numpy()
    std_labels, label_map = _standardise_labels(raw_labels, x_comp)
    wrapper = GMMWrapper(gmm, x_comp, x_features)
    xs = np.linspace(*xrange, resolution)
    ys = np.linspace(*yrange, resolution)
    xx, yy = np.meshgrid(xs, ys)
    z_raw = wrapper.predict(np.c_[xx.ravel(), yy.ravel()])
    z = np.vectorize(label_map.get)(z_raw).reshape(xx.shape)
    
    if region_colors is None:
        region_colors = ["aquamarine", "lightsteelblue"]
    
    traces: List[Contour] = []
    if plot_regions:
        discrete_colorscale = [
            [0.0, region_colors[0]], 
            [0.499, region_colors[0]],
            [0.5, region_colors[1]], 
            [1.0, region_colors[1]]
        ]
        traces.append(go.Contour(
            x=xs, 
            y=ys, 
            z=z,
            showscale=False, 
            opacity=decision_alpha,
            colorscale=discrete_colorscale, 
            contours=dict(coloring="fill"),
            hoverinfo="skip", 
            name="Phase Regions", 
            showlegend=False
        ))

    contour_line = go.Contour(
        x=xs, 
        y=ys, 
        z=z,
        showscale=False,
        colorscale=[[0, boundary_color], 
                    [1, boundary_color]],
        zmin=0,
        zmax=1,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none",
            showlines=True
        ),
        line=dict(color="Red", width=3),
        name="<b>Decision Bounday (Experiment)</b>", 
        hoverinfo="skip"
    )
    traces.append(contour_line)

    boundary_points = extract_boundary_from_contour(z, 
                                                    xs, 
                                                    ys,
                                                    level=0.5)
    return gmm, std_labels, boundary_points, traces


def plot_gmm_composition_phase(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange=List[int],
    yrange=List[int],    
    n_components: int = 2,
    random_state: int = 42,
    region_colors: Optional[List[str]] = None,
    resolution: int = 200,
    plot_regions: bool = True,
    point_cmap: Optional[List[str]] = None
) -> Tuple[GaussianMixture, np.ndarray, Optional[np.ndarray], List[Union[Contour, Scatter]]]:
    """
    Trains a GMM and creates a scatter trace for data points.

    Optionally creates region traces as well, mimicking the 
    original script's logic.

    Parameters:
    -----------
        df (pd.DataFrame): DataFrame with composition and 
                           phase data.
        x_col (str): Name of the column for the x-axis 
                     component.
        y_col (str): Name of the column for the y-axis 
                     component.
        phase_col (str): Name of the column containing 
                         phase information.
        xrange (List[int]): X-axis composition range.
        yrange (List[int]): Y-axis composition range. 
        n_components (int): Number of GMM components.
        random_state (int): Seed for reproducibility.
        region_colors (Optional[List[str]]): Colors for the 
                                             phase regions.
        resolution (int): Grid resolution for any contour plots.
        plot_regions (bool): Whether to plot the filled regions.
        point_cmap (Optional[List[str]]): Colormap for the
                                          scatter points.

    Returns:
    -------
        Tuple: Containing the GMM model, cluster labels, 
               boundary points, and a
        list of Plotly traces (Contour and/or Scatter).
    """
    df_local = df.copy()
    x_features = df_local[phase_col].to_numpy().reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state
    )
    raw_labels = gmm.fit_predict(x_features)

    x_comp = df_local[[x_col, y_col]].to_numpy()
    std_labels, label_map = _standardise_labels(raw_labels, x_comp)
    wrapper = GMMWrapper(gmm, x_comp, x_features)
    xs = np.linspace(*xrange, resolution)
    ys = np.linspace(*yrange, resolution)
    xx, yy = np.meshgrid(xs, ys)
    z_raw = wrapper.predict(np.c_[xx.ravel(), yy.ravel()])
    z = np.vectorize(label_map.get)(z_raw).reshape(xx.shape)

    traces: List[Union[Contour, Scatter]] = []
    if plot_regions and region_colors is not None:
        discrete = [
            [0.0, region_colors[0]],
            [0.499, region_colors[0]],
            [0.5, region_colors[1]],
            [1.0, region_colors[1]],
        ]
        traces.append(
            go.Contour(
                x=xs,
                y=ys,
                z=z,
                showscale=False,
                colorscale=discrete,
                contours=dict(coloring="fill"),
                hoverinfo="skip",
                name="Calculated Regions",
                showlegend=False,
            )
        )

    symbols = ["triangle-up" if lbl == 0 else "square" for lbl in std_labels]
    if point_cmap is not None:
        colours = [point_cmap[lbl] for lbl in std_labels]
    elif region_colors is not None:
        colours = [region_colors[lbl] for lbl in std_labels]
    else:
        colours = ["blue"] * len(std_labels)

    traces.append(
        go.Scatter(
            x=x_comp[:, 0],
            y=x_comp[:, 1],
            mode="markers",
            marker=dict(
                size=12,
                color=colours,
                symbol=symbols,
                line=dict(width=1, color="black"),
            ),
            customdata=df_local[phase_col].values,
            hovertemplate=(
                "Feature Value: %{customdata}<br>X: %{x}<br>Y: %{y}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    boundary = extract_boundary_from_contour(z, xs, ys, level=0.5)
    return gmm, std_labels, boundary, traces

def titration_diagram(
    file_path: str,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange: List[int],
    yrange: List[int],
    output_path: str,
) -> None:
    """
    Load data and plot two-phase scatter diagram.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file.
    x_col : str
        Name of the column for x-axis values.
    y_col : str
        Name of the column for y-axis values.
    phase_col : str
        Column name for phase labels (0 or 1).
    xrange : List[int]
        [min, max] range for the x-axis.
    yrange : List[int]
        [min, max] range for the y-axis.
    output_path : str
        Path to save the output PNG image.

    Returns
    -------
    None
        Displays the plot and writes image to output_path.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.",
              file=sys.stderr)
        sys.exit(1)

    fig = go.Figure()

    single = df[df[phase_col] == 0]
    fig.add_trace(go.Scatter(
        x=single[x_col],
        y=single[y_col],
        mode="markers",
        marker=dict(
            size=12,
            color="dodgerblue",
            symbol="square",
            line=dict(width=1, color="black"),
        ),
        name="Single Phase",
    ))

    two = df[df[phase_col] == 1]
    fig.add_trace(go.Scatter(
        x=two[x_col],
        y=two[y_col],
        mode="markers",
        marker=dict(
            size=12,
            color="#FFFFCC",
            symbol="triangle-up",
            line=dict(width=1, color="black"),
        ),
        name="Two Phase",
    ))

    fig.update_layout(
        xaxis_title=f"<b>{x_col}</b>",
        yaxis_title=f"<b>{y_col}</b>",
        margin=dict(l=60, r=60, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.8, y=0.8,
                    bgcolor="rgba(255,255,255,0.8)"),
        width=800,
        height=800,
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(font=dict(size=24)),
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            title=dict(font=dict(size=24)),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24),
        ),
    )
    fig.update_xaxes(mirror="ticks", range=xrange)
    fig.update_yaxes(mirror="ticks", range=yrange)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.show()
    fig.write_image(output_path)
    print(f"Plot saved successfully to {output_path}")

def plot_two_scatter(
    excel_file: str,
    sheet1: str,
    sheet2: str,
    output_path: str,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
) -> None:
    """
    Read two Excel sheets and plot styled scatter comparison.

    Parameters
    ----------
    excel_file : str
        Path to Excel file containing both sheets.
    sheet1 : str
        Name of the first sheet.
    sheet2 : str
        Name of the second sheet.
    output_path : str
        Path to save the output image.
    xlim : Optional[List[float]]
        [min, max] range for x-axis.
    ylim : Optional[List[float]]
        [min, max] range for y-axis.

    Returns
    -------
    None
        Displays the plot and writes image to output_path.
    """

    try:
        xls = pd.ExcelFile(excel_file)
    except FileNotFoundError:
        print(
            f"Error: The file '{excel_file}' was not found.",
            file=sys.stderr
        )
        sys.exit(1)

    missing = [s for s in (sheet1, sheet2) if s not in xls.sheet_names]
    
    if missing:
        print(
            f"Error: Sheet(s) {missing} not found in '{excel_file}'.",
            file=sys.stderr
        )
        sys.exit(1)
    
    df1 = xls.parse(sheet1)
    df2 = xls.parse(sheet2)

    x_col = list(df2.columns)[0]
    y_col = list(df2.columns)[1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df1[x_col],
        y=df1[y_col],
        mode="markers",
        marker=dict(
            size=12,
            color="purple",
            symbol="triangle-up",
            line=dict(width=1, color="black")
        ),
        name="Titration"
    ))

    fig.add_trace(go.Scatter(
        x=df2[x_col],
        y=df2[y_col],
        mode="markers",
        marker=dict(
            size=12,
            color="yellow",
            symbol="triangle-up",
            line=dict(width=1, color="black")
        ),
        name="Our Method"
    ))

    fig.update_layout(
        xaxis_title=f"<b>{x_col}</b>",
        yaxis_title=f"<b>{y_col}</b>",
        margin=dict(l=60, r=60, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.80, y=0.90,
                    bgcolor="rgba(255,255,255,0.8)"),
        width=800,
        height=800
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(font=dict(size=24)),
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24)
        ),
        yaxis=dict(
            title=dict(font=dict(size=24)),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24)
        )
    )
    fig.update_xaxes(mirror="ticks", range=xlim)
    fig.update_yaxes(mirror="ticks", range=ylim)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.show()
    fig.write_image(output_path)
    print(f"Plot saved successfully to {output_path}")

def mathematical_model(
    file_path: str,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange: List[int],
    yrange: List[int],
    output_path: str,
):
    """
    Generate and save a phase diagram plot from composition data.

    Parameters
    ----------
    file_path : str
        Path to the CSV input file.
    x_col : str
        Column name for x-axis values.
    y_col : str
        Column name for y-axis values.
    phase_col : str
        Column name for phase labels.
    xrange : List[int]
        Two-element list [min, max] for x-axis range.
    yrange : List[int]
        Two-element list [min, max] for y-axis range.
    output_path : str
        Path to save the output plot image.

    Returns
    -------
    None
        Displays the plot and writes the image to output_path.
    """
    MODEL_A = 0.955
    MODEL_DA = 0.028
    MODEL_B = -5.73
    MODEL_DB = 0.17
    MODEL_C = 581
    MODEL_DC = 29
    MODEL_R2 = 0.9993

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    fig = go.Figure()

    _, _, _, traces_exp = plot_gmm_decision_regions(
        df=df,
        x_col=x_col,
        y_col=y_col,
        phase_col=phase_col,
        xrange=xrange,
        yrange=yrange,
        n_components=2,
        random_state=42,
        region_colors=["aquamarine", "lightsteelblue"],
        boundary_color="red",
        resolution=200,
        decision_alpha=1,
        plot_regions=True,
    )
    for t in traces_exp:
        fig.add_trace(t)

    _, _, _, traces_act = plot_gmm_composition_phase(
        df=df,
        x_col=x_col,
        y_col=y_col,
        phase_col=phase_col,
        xrange=xrange,
        yrange=yrange,
        n_components=2,
        random_state=42,
        region_colors=None,
        plot_regions=False,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    for t in traces_act:
        fig.add_trace(t)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=20, color="lightsteelblue", symbol="square"),
            name="<b>Single Phase (Experiment)</b>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=20, color="aquamarine", symbol="square"),
            name="<b>Two Phase (Experiment)</b>",
        )
    )

    fig.update_layout(
        xaxis_title=f"<b>{x_col}</b>",
        yaxis_title=f"<b>{y_col}</b>",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=1.0, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        width=800,
        height=800,
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(font=dict(size=24)),
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            title=dict(font=dict(size=24)),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24),
        ),
    )
    fig.update_xaxes(mirror="ticks", range=xrange)
    fig.update_yaxes(mirror="ticks", range=yrange)

    raw_X = np.linspace(xrange[0], xrange[1], 500)
    frac_X = raw_X / 100.0

    params = {
        "Min": (MODEL_A - MODEL_DA, MODEL_B - MODEL_DB, MODEL_C - MODEL_DC),
        "Mean": (MODEL_A, MODEL_B, MODEL_C),
        "Max": (MODEL_A + MODEL_DA, MODEL_B + MODEL_DB, MODEL_C + MODEL_DC),
    }
    line_styles = {"Min": "dot", "Mean": "solid", "Max": "dash"}

    for label, (a, b, c) in params.items():
        y_frac = a * np.exp(b * np.sqrt(frac_X) - c * frac_X**3)
        model_y = y_frac * 100.0
        fig.add_trace(go.Scatter(
            x=raw_X,
            y=model_y,
            mode="lines",
            line=dict(color="black", dash=line_styles[label]),
            name=f"Model ({label}): a={a:.3f}, b={b:.2f}, c={c:.0f}",
        ))

    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.9,
        text=f"R² = {MODEL_R2:.4f}", showarrow=False,
        font=dict(size=14),
    )

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.show()
    fig.write_image(output_path)
    print(f"Plot saved successfully to {output_path}")

def phase_diagram_exp(
    file_path: str,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange: List[float],
    yrange: List[float],
    output_path: str,
) -> None:
    """
    Load data, generate the phase diagram, and write the image.

    Parameters
    ----------
    file_path :
        Path to a CSV file with the required columns.
    x_col, y_col :
        Column names for the two composition axes.
    phase_col :
        Column holding the phase indicator used to fit the GMM.
    xrange, yrange :
        Two-element lists specifying axis extents.
    output_path :
        Destination filename (PNG, SVG, etc.) for the saved figure.

    Returns
    -------
    None
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    fig = go.Figure()

    _, _, _, traces_exp = plot_gmm_decision_regions(
        df=df,
        x_col=x_col,
        y_col=y_col,
        phase_col=phase_col,
        xrange=xrange,
        yrange=yrange,
        n_components=2,
        random_state=42,
        region_colors=["aquamarine", "lightsteelblue"],
        boundary_color="red",
        resolution=200,
        decision_alpha=1,
        plot_regions=True,
    )
    for t in traces_exp:
        fig.add_trace(t)

    _, _, _, traces_act = plot_gmm_composition_phase(
        df=df,
        x_col=x_col,
        y_col=y_col,
        phase_col=phase_col,
        xrange=xrange,
        yrange=yrange,
        n_components=2,
        random_state=42,
        region_colors=None,
        plot_regions=False,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    for t in traces_act:
        fig.add_trace(t)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=20, color="lightsteelblue", symbol="square"),
            name="<b>Single Phase (Experiment)</b>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=20, color="aquamarine", symbol="square"),
            name="<b>Two Phase (Experiment)</b>",
        )
    )

    fig.update_layout(
        xaxis_title=f"<b>{x_col}</b>",
        yaxis_title=f"<b>{y_col}</b>",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=1.0, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        width=800,
        height=800,
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(font=dict(size=24)),
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            title=dict(font=dict(size=24)),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            domain=[0, 1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="inside",
            ticklen=5,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=24),
        ),
    )
    fig.update_xaxes(mirror="ticks", range=xrange)
    fig.update_yaxes(mirror="ticks", range=yrange)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.show()
    fig.write_image(output_path)
    print(f"Plot saved to '{output_path}'.")

if __name__ == "__main__":
    ## Create the dataset relevant for phase diagram based on
    ## titration and plot the phase diagram 
    read_composition_excel(
        './data/Figure_3_Data.xlsx', 
        'Figure_3_PEO8k_Sodium_Citrate',
        'Sodium Citrate (wt%)',
        'PEO 8 kg/mol (wt%)',
        'Phase_Separation',
        './data/Figure_3_PEO8K_Sodium_Citrate.csv'
    )

    titration_diagram(
        file_path='./data/Figure_3_PEO8K_Sodium_Citrate.csv',
        x_col='Sodium Citrate (wt%)',
        y_col='PEO 8 kg/mol (wt%)',
        phase_col='Phase_Separation',
        xrange=[0, 20],
        yrange=[0, 35],
        output_path='./Manuscript_Figures/' \
        'Figure_3_PEO8K_Sodium_Citrate_Titration_Phase_Diagram_With_No_Regions.png'
    )

    read_composition_excel(
        './data/Figure_3_Data.xlsx', 
        'Figure_3_PEO10K_DEX10K',
        'Dextran 10 kg/mol (wt%)',
        'PEO 10 kg/mol (wt%)',
        'Phase_Separation',
        './data/Figure_3_PEO10K_DEX10K.csv'
    )

    titration_diagram(
        file_path='./data/Figure_3_PEO10K_DEX10K.csv',
        x_col='Dextran 10 kg/mol (wt%)',
        y_col='PEO 10 kg/mol (wt%)',
        phase_col='Phase_Separation',
        xrange=[0, 13],
        yrange=[0, 13],
        output_path='./Manuscript_Figures/' \
        'Figure_3_PEO10K_DEX10K_Titration_Phase_Diagram_With_No_Regions.png'
    )

    read_composition_excel(
        './data/Figure_3_Data.xlsx', 
        'Figure_3_PEO20K_DEX500K',
        'Dextran 500 kg/mol (wt%)',
        'PEO 20 kg/mol (wt%)',
        'Phase_Separation',
        './data/Figure_3_PEO20K_DEX500K.csv'
    )

    titration_diagram(
        file_path='./data/Figure_3_PEO20K_DEX500K.csv',
        x_col='Dextran 500 kg/mol (wt%)',
        y_col='PEO 20 kg/mol (wt%)',
        phase_col='Phase_Separation',
        xrange=[0, 10],
        yrange=[0, 6],
        output_path='./Manuscript_Figures/' \
        'Figure_3_PEO20K_DEX500K_Titration_Phase_Diagram_With_No_Regions.png'
    )

    ## Comparison of decision boundaries (binodal curves)
    ## generated by titration-based method and microscopy.
    plot_two_scatter(
        excel_file='./data/Figure_6_Data.xlsx',
        sheet1='Figure6_Titrate_PEO20K_DEX500K',
        sheet2='Figure6_TECAN_PEO20K_DEX500K',
        output_path='./Manuscript_Figures/' \
        'Figure_6_PEO20K_DEX500K_Binodal_Comparison.png',
        xlim=[0, 11],
        ylim=[0, 4],
    )

    plot_two_scatter(
        excel_file='./data/Figure_6_Data.xlsx',
        sheet1='Figure6_Titrate_PEO8K_Sod',
        sheet2='Figure6_TECAN_PEO8K_Sod',
        output_path='./Manuscript_Figures/' \
        'Figure_6_PEO8K_Sodium_Citrate_Binodal_Comparison.png',
        xlim=[0, 20],
        ylim=[0, 40],
    )

    plot_two_scatter(
        excel_file='./data/Figure_6_Data.xlsx',
        sheet1='Figure6_Titrate_PEO10K_DEX10K',
        sheet2='Figure6_TECAN_PEO10K_DEX10K',
        output_path='./Manuscript_Figures/' \
        'Figure_6_PEO10K_DEX10K_Binodal_Comparison.png',
        xlim=[0, 13],
        ylim=[0, 13]
    )                                         

    ## Overlap the mathematical model showing the binodal curve
    ## of PEO8K-Sodium Citrate system. 
    ## Cite:
        # Silvério, Sara C., et al. "Effect of aqueous two-phase
        # system constituents in different poly (ethylene glycol)–
        # salt phase diagrams." Journal of Chemical & Engineering
        # Data 57.4 (2012): 1203-1208.
    mathematical_model(
        file_path='./data/' \
        'PEO8K_Sodium_Citrate_Composition_Phase.csv',
        x_col='Sodium Citrate (wt%)',
        y_col='PEO 8 kg/mol (wt%)',
        phase_col='Phase_Separation_2nd',
        xrange=[0, 21],
        yrange=[0, 38],
        output_path='./Manuscript_Figures/' \
        'PEO8K_Sodium_Citrate_Phase_Diagram_Experiment_2nd_Time.png'
    )

    ## Phase Diagram of microscopy experiments provided by
    ## experiment team. 
    phase_diagram_exp(
        file_path='./data/' \
        'PEO20K_DEX500K_Composition_Phase.csv',
        x_col='Dextran 500 kg/mol (wt%)',
        y_col='PEO 20 kg/mol (wt%)',
        phase_col='Phase_Separation_1st',
        xrange=[0, 11],
        yrange=[0, 4],
        output_path='./Manuscript_Figures/'
        'PEO20K_DEX500K_Phase_Diagram_Experiment_1st_Time.png'
    )

    phase_diagram_exp(
        file_path='./data/' \
        'PEO20K_DEX500K_Composition_Phase.csv',
        x_col='Dextran 500 kg/mol (wt%)',
        y_col='PEO 20 kg/mol (wt%)',
        phase_col='Phase_Separation_2nd',
        xrange=[0, 11],
        yrange=[0, 4],
        output_path='./Manuscript_Figures/'
        'PEO20K_DEX500K_Phase_Diagram_Experiment_2nd_Time.png'
    )

    phase_diagram_exp(
        file_path='./data/' \
        'PEO10K_DEX10K_Composition_Phase.csv',
        x_col='Dextran 10 kg/mol (wt%)',
        y_col='PEO 10 kg/mol (wt%)',
        phase_col='Phase_Separation_1st',
        xrange=[0, 13],
        yrange=[0, 13],
        output_path='./Manuscript_Figures/'
        'PEO10K_DEX10K_Phase_Diagram_Experiment_1st_Time.png'
    )

    phase_diagram_exp(
        file_path='./data/' \
        'PEO10K_DEX10K_Composition_Phase.csv',
        x_col='Dextran 10 kg/mol (wt%)',
        y_col='PEO 10 kg/mol (wt%)',
        phase_col='Phase_Separation_2nd',
        xrange=[0, 13],
        yrange=[0, 13],
        output_path='./Manuscript_Figures/'
        'PEO10K_DEX10K_Phase_Diagram_Experiment_2nd_Time.png'
    )

    phase_diagram_exp(
        file_path='./data/' \
        'PEO8K_Sodium_Citrate_Composition_Phase.csv',
        x_col='Sodium Citrate (wt%)',
        y_col='PEO 8 kg/mol (wt%)',
        phase_col='Phase_Separation_1st',
        xrange=[0, 21],
        yrange=[0, 40],
        output_path='./Manuscript_Figures/'
        'PEO8K_Sodium_Citrate_Phase_Diagram_Experiment_1st_Time.png'
    )

    phase_diagram_exp(
        file_path='./data/' \
        'PEO8K_Sodium_Citrate_Composition_Phase.csv',
        x_col='Sodium Citrate (wt%)',
        y_col='PEO 8 kg/mol (wt%)',
        phase_col='Phase_Separation_2nd',
        xrange=[0, 21],
        yrange=[0, 40],
        output_path='./Manuscript_Figures/'
        'PEO8K_Sodium_Citrate_Phase_Diagram_Experiment_2nd_Time.png'
    )