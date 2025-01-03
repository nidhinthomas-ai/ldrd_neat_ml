from neat_ml import lib
import os
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.ensemble import RandomForestClassifier
from matplotlib.testing.compare import compare_images


def test_cesar_cg_rdf_labels():
    # from: https://gitlab.lanl.gov/treddy/ldrd_neat_ml/-/issues/25
    # ensure Cesar's CG-MD RDF data columns have names
    # that are more appropriate than unlabelled floating point
    # values
    actual_df_cols = lib.read_in_cesar_cg_md_data().columns
    for col in actual_df_cols[10:]:
        assert str(col).startswith("CG_RDF_")


@pytest.mark.parametrize("df, expected_prefix", [
    (lib.read_in_cesar_cg_md_data(), "CG_"),
    (lib.read_in_cesar_all_atom_md_data(), "AA_"),
    ],
)
def test_cesar_data_column_prefix_by_type(df, expected_prefix):
    # ensure that the CG and AA MD data column
    # titles are prefixed with a string that
    # clearly distinguished the two types of simulations
    # (helpful for i.e., feature importance analysis)
    for col_name in df.columns[3:]:
        assert str(col_name).startswith(expected_prefix)


@pytest.mark.parametrize("pos_shap_vals, feature_names, top_feat_count, expected_names, expected_counts", [
    ([# first two features important for this model
      np.array([[0.9, 0.7, 0.0],
               [0.8, 0.6, 0.1]]),
      # last two features important for this model
      np.array([[0.0, 0.8, 0.6],
               [0.1, 0.8, 0.7]])],
     np.asarray([f"Feat_{i}" for i in range(3)]),
     2,
    # Feat_1 is of overlapping importance
    # so should show up as important for both;
    # the other two features should show up as
    # important once each
    ["Feat_1", "Feat_0", "Feat_2"],
    [2, 1, 1],
    ),
     # similar case but with an extra column + extra model
    ([# first two and last features important for this model
      np.array([[0.9, 0.7, 0.0, 0.7],
               [0.8, 0.6, 0.1, 0.7]]),
      # first and last two feature important in this model
      np.array([[0.8, 0.1, 0.2, 0.9],
               [0.8, 0.1, 0.8, 0.9]]),
      # first three features important
      np.array([[0.7, 0.7, 0.7, 0.1],
                [0.7, 0.7, 0.8, 0.1]])],
     np.asarray([f"Feat_{i}" for i in range(4)]),
     3,
    ["Feat_0", "Feat_3", "Feat_1", "Feat_2"],
    [3, 2, 2, 2],
    ),
    # for cases when i.e., SHAP and non-SHAP feature
    # importances are combined, SHAP may require the mean
    # reduction operation but non-SHAP may already be reduced
    # across all records
    ([# first two features important for this model (SHAP-style)
      np.array([[0.9, 0.7, 0.0],
                [0.8, 0.6, 0.1]]),
      # last two features important for this model (reduced style,
      # like native Random Forest feature importances)
      np.array([0.0, 0.8, 0.6])],
      np.asarray([f"Feat_{i}" for i in range(3)]),
      2,
    # Feat_1 is of overlapping importance
    # so should show up as important for both;
    # the other two features should show up as
    # important once each
    ["Feat_1", "Feat_0", "Feat_2"],
    [2, 1, 1],
    ),

    ])
def test_feat_import_consensus(pos_shap_vals,
                               feature_names,
                               top_feat_count,
                               expected_names,
                               expected_counts,
                               ):
    (actual_ranked_feat_names,
     actual_ranked_feat_counts,
     num_input_models) = lib.feature_importance_consensus(pos_shap_vals,
                                                          feature_names,
                                                          top_feat_count)
    assert num_input_models == len(pos_shap_vals)
    assert_array_equal(actual_ranked_feat_names, expected_names)
    assert_array_equal(actual_ranked_feat_counts, expected_counts)


def test_plot_feat_import_consensus(tmp_path):
    # crude regression test for plot_feat_import_consensus();
    # just a few sanity/smoke checks...
    ranked_feature_names = np.asarray([f"feat_{i}" for i in range(4)])
    ranked_feature_counts = np.asarray([4, 3, 2, 2])
    num_input_models = 4
    top_feat_count = 3
    fig_name = "tmp_feat_imp_consensus.png"
    tmp_fig = tmp_path / fig_name
    actual_fig = lib.plot_feat_import_consensus(ranked_feature_names,
                                                ranked_feature_counts,
                                                num_input_models,
                                                top_feat_count,
                                                tmp_fig)
    # check that the plot file was produced:
    assert tmp_fig.exists()
    # check that the bar at the bottom is the largest
    axis = actual_fig.get_axes()[0]
    actual_patches = axis.patches
    actual_bar_widths = []
    for patch in actual_patches:
        actual_bar_widths.append(patch.get_width())
    assert np.argmax(actual_bar_widths) == 0
    assert len(actual_bar_widths) == ranked_feature_counts.size


@pytest.mark.parametrize("important_col, expected", [
    # should be easy to predict the important
    # feature for the non-zero rows
    (0, [0, 0, 0]),
    (1, [1, 1, 1]),
    (2, [2, 2, 2]),
])
def test_build_lime_data(important_col, expected):
    # very simple dataset to check the most basic
    # properties of the LIME data structure generation
    data = np.zeros(shape=(4, 3), dtype=np.float64)
    data[:-1, important_col] = 1.0
    y = np.asarray([1, 1, 1, 0])
    X = pd.DataFrame(data=data, columns=["1", "2", "3"])
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X, y)
    lime_actual = lib.build_lime_data(X=X, model=rf)
    assert lime_actual.shape == X.shape
    # the last row has no "signal" so is harder
    # to predict LIME value for, but the score
    # of the important_col should be negative I think
    # because of an absence of value there
    assert_array_equal(lime_actual.argmax(axis=1)[:-1], expected)
    assert lime_actual[-1].argmax() != important_col


def test_build_lime_row_selection():
    # regression test for:
    # https://gitlab.lanl.gov/treddy/ldrd_neat_ml/-/merge_requests/36#note_279748
    data = np.eye(3, k=1)
    y = np.asarray([1, 1, 0])
    X = pd.DataFrame(data=data, columns=["1", "2", "3"])
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X, y)
    lime_actual = lib.build_lime_data(X=X, model=rf)
    assert lime_actual.shape == X.shape
    assert_array_equal(lime_actual.argmax(axis=1), [1, 2, 0])


def test_plot_ebm_data_non_interacting(tmp_path):
    # plot_ebm_data didn't originally support
    # non-interacting top features
    scores = np.asarray([0.8, 0.3])
    names = np.asarray(["feature_0000", "feature_0001"])
    fig_title = "Test"
    fig_name = "test.png"
    tmp_fig = tmp_path / fig_name
    explain_data = {"scores": scores,
                    "names": names}
    # for now, simply check for absence of error:
    lib.plot_ebm_data(explain_data=explain_data,
                      original_feat_names=["feat_1", "feat_2"],
                      fig_title=fig_title,
                      fig_name=tmp_fig,
                      top_feat_count=2)
    
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Create a small sample dataframe for testing.
    This includes columns for 'WT% DEX', 'WT% PEO',
    and 'median_radii'.
    """
    data = {
        "WT% DEX": [0, 0, 5, 10, 15],
        "WT% PEO": [0, 5, 5, 10, 15],
        "median_radii": [0, 0, 0, 1.5, 1.5],
    }
    return pd.DataFrame(data)

@pytest.mark.parametrize("method", ["nearest"])
def test_interpolate_and_plot(sample_df, tmp_path, method) -> None:
    """
    Test the interpolate_and_plot function with
    a small DataFrame and verify that it runs
    without errors and creates output files for
    each interpolation method.
    """
    fig_name_prefix = "interp_"
    fig_prefix = tmp_path / fig_name_prefix
    reference_dir = os.path.join(os.path.dirname(__file__), "baseline")
    lib.interpolate_and_plot(
        df=sample_df,
        x_col="WT% DEX",
        y_col="WT% PEO",
        value_col="median_radii",
        methods=[method],
        x_min=0,
        x_max=15,
        y_min=0,
        y_max=15,
        grid_points=10j,
        fig_prefix=fig_prefix
    )
    # Check if the image is comparable to reference images
    reference_image = os.path.join(reference_dir, f"interp_{method}_ref.png")
    nearest_file = str(fig_prefix) + f"{method}.png"
    diff = compare_images(reference_image, nearest_file, tol=2)
    assert diff is None, f"Images do not match for method {method}: {diff}"

@pytest.mark.parametrize("kernel_method", ["rbf", "linear"])
def test_svc_classification_and_plot(sample_df, tmp_path, kernel_method)-> None:
    """
    Test the svc_classification_and_plot function with a small DataFrame and verify that
    it runs without errors and creates output files for each SVC kernel method.
    """
    fig_name_prefix = "hyper_"
    fig_prefix = tmp_path / fig_name_prefix
    reference_dir = os.path.join(os.path.dirname(__file__), "baseline")

    # Since we have few points, not all kernels may be meaningful, 
    # but we just need to ensure the code runs.
    lib.svc_classification_and_plot(
        df=sample_df,
        x_col="WT% DEX",
        y_col="WT% PEO",
        value_col="median_radii",
        threshold=1,
        methods=[kernel_method],
        gamma="scale",
        fig_prefix=fig_prefix
    )

    # Check if the images are comparable to reference images
    ref_image = os.path.join(reference_dir, f"hyper_{kernel_method}_ref.png")
    gen_image = str(fig_prefix) + f"{kernel_method}.png"
    diff = compare_images(ref_image, gen_image, tol=2)
    assert diff is None, (
        f"Images do not match for kernel '{kernel_method}': {diff}"
        )
