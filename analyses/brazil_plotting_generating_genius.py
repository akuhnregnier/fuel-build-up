# -*- coding: utf-8 -*-
import logging
import os

import cartopy.crs as ccrs
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Also instruct autoflake to keep this line.
import nc_time_axis  # noqa
from wildfires.analysis import *
from wildfires.data import *
from wildfires.logging_config import enable_logging
from wildfires.utils import get_land_mask

logger = logging.getLogger(__name__)
enable_logging()

memory = get_memory("analysis_brazil_generating_genius")

# This is where the reserve is located.
reserve_lon = -55.629804
reserve_lat = -3.170176


@memory.cache()
def get_temporal_variables(
    dataset_names,
    variable_names,
    reserve_lats=(reserve_lat - 1, reserve_lat + 1),
    reserve_lons=(reserve_lon - 1, reserve_lon + 1),
    min_year=2000,
):
    """Return selected variables temporal evolution in a region around the reserve.

    Args:
        dataset_names (iterable of str): Datasets to select.
        variable_names (iterable of str): Variables to select. Must be present in the
            datasets referred to by `dataset_names`.

    Returns:
        Datasets: Spatially averaged variables.

    """
    import wildfires.data

    datasets = Datasets()
    for dataset_name in dataset_names:
        datasets.add(getattr(wildfires.data, dataset_name)())

    # Select the requested variables from the datasets.
    datasets.select_variables(variable_names)

    # Average the cubes in the reserve area.
    for dataset in datasets:
        for i, cube in enumerate(dataset):
            if not cube.coord("latitude").has_bounds():
                cube.coord("latitude").guess_bounds()
            if not cube.coord("longitude").has_bounds():
                cube.coord("longitude").guess_bounds()
            cube = cube.extract(
                iris.Constraint(
                    coord_values=dict(
                        time=lambda cell: min_year <= cell.point.year,
                        latitude=lambda cell: reserve_lats[0] < cell < reserve_lats[1],
                        longitude=lambda cell: reserve_lons[0] < cell < reserve_lons[1],
                    )
                )
            )
            dataset.cubes[i] = cube.collapsed(
                ("latitude", "longitude"),
                iris.analysis.MEAN,
                weights=iris.analysis.cartography.area_weights(cube),
            )
    # Carry out the calculation by realising data.
    datasets.cubes.realise_data()
    return datasets


# Get mean BA.
@memory.cache()
def get_avg_ba():
    avg_ba = MCD64CMQ_C6().cube.collapsed("time", iris.analysis.MEAN)
    # Realise the data.
    _ = avg_ba.data
    return avg_ba


def plot_burned_area_maps(figure_saver):
    """Plot burned area globally, and with a focus on the reserve."""
    avg_ba = get_avg_ba()

    # Apply land mask.
    avg_ba.data.mask |= ~get_land_mask()

    # Ignore 0 BA.
    avg_ba.data.mask |= np.isclose(avg_ba.data, 0)

    brazil_lats = (-50, 10)
    brazil_lons = (-80, -30)

    # Plot Global BA, with one plot zooming in on Brazil.
    mpl.rc("figure", figsize=(7.26, 4.54))

    with figure_saver("global_ba"):
        fig = plt.figure()
        top = 0.9
        axes = (
            fig.add_axes([0, 0, 0.7, top], projection=ccrs.Robinson()),
            fig.add_axes(
                [0.665, 0, 0.25, top],
                projection=ccrs.Robinson(central_longitude=np.mean(brazil_lons)),
            ),
        )
        _, _, img, _ = cube_plotting(
            avg_ba,
            cmap="brewer_RdYlBu_11_r",
            ax=axes[0],
            log=True,
            orientation="horizontal",
            coastline_kwargs={"linewidth": 0.5},
            animation_output=True,
            new_colorbar=False,
            title=None,
        )
        cube_plotting(
            avg_ba.extract(
                iris.Constraint(
                    coord_values=dict(
                        latitude=lambda cell: brazil_lats[0] < cell < brazil_lats[1],
                        longitude=lambda cell: brazil_lons[0] < cell < brazil_lons[1],
                    )
                )
            ),
            cmap="brewer_RdYlBu_11_r",
            ax=axes[1],
            log=True,
            orientation="horizontal",
            coastline_kwargs={"linewidth": 0.5},
            new_colorbar=False,
            title=None,
        )

        # fig.suptitle = "Burned Area (MCD64CMQ Collection 6)"

        # Create a colorbar based on the global BA data.
        fig.colorbar(
            img,
            label="Average Burned Area Fraction (1)",
            orientation="horizontal",
            fraction=0.38,
            pad=0.07,
            shrink=0.7,
            aspect=50,
            anchor=(0.5, 1.0),
            panchor=(0.5, 1.0),
            format="%0.1e",
            ax=axes,
        )
        # Indicate the location of the Tapajós-Arapiuns Extractive Reserve.
        mc = "#2ede02"
        # mc = "#f568f5"
        for ax, kwargs in zip(
            axes,
            ({}, {"markeredgewidth": 3, "markerfacecolor": "none", "markersize": 13}),
        ):
            ax.plot(
                reserve_lon,
                reserve_lat,
                marker="o",
                transform=ccrs.PlateCarree(),
                markersize=kwargs.pop("markersize", 3),
                markerfacecolor=kwargs.pop("markerfacecolor", mc),
                markeredgecolor=kwargs.pop("markerfacecolor", mc),
                **kwargs,
            )

        # Add an arrow to indicate the location.
        axes[1].annotate(
            "Reserve",
            xy=(0.54, 0.785),
            xytext=(0.6, 0.9),
            arrowprops=dict(arrowstyle="simple"),
            xycoords="axes fraction",
        )
    plt.close()


if __name__ == "__main__":
    # Workaround to allow plotting of cftime dates.

    figure_saver = FigureSaver(
        directories=os.path.join(os.path.expanduser("~"), "tmp", "generating_genius"),
        debug=True,
        dpi=500,
    )

    # plot_burned_area_maps(figure_saver)

    # Plot a series of variables
    datasets = get_temporal_variables(
        (
            "ERA5_CAPEPrecip",
            "ERA5_DryDayPeriod",
            "ERA5_Temperature",
            "HYDE",
            "MCD64CMQ_C6",
        ),
        (
            "Burned Area",
            "CAPE x Precip",
            "Dry Day Period",
            "Max Temp",
            "Mean Temp",
            "popd",
            "cropland",
        ),
    )
    plot_data = {}
    times = {}
    data = {}
    dataset_names = {}

    for dataset_name, dataset in zip(datasets.pretty_dataset_names, datasets):
        for variable_name, cube in zip(dataset.variable_names("pretty"), dataset):
            plot_data[variable_name] = {
                "times": cube.coord("time").units.num2date(cube.coord("time").points),
                "data": cube.data,
                "dataset_name": dataset_name,
            }
    with figure_saver("individual_variables"):
        fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 10))

        var = "MCD64CMQ BA"
        ax = axes[0, 0]
        ax.plot(
            plot_data[var]["times"], plot_data[var]["data"],
        )
        ax.set_title("MCD64CMQ")
        ax.set_ylabel("Burned Area Fraction")
        ax.set_yscale("log")

        var = "CAPE x Precip"
        ax = axes[0, 1]
        ax.plot(
            plot_data[var]["times"], plot_data[var]["data"],
        )
        ax.set_title("ERA5")
        ax.set_ylabel(var)

        var = "Dry Day Period"
        ax = axes[1, 0]
        ax.plot(
            plot_data[var]["times"], plot_data[var]["data"],
        )
        ax.set_title("ERA5")
        ax.set_ylabel("Dry Day Period (days)")
        # ax.set_yscale('log')

        var = "Max Temp"
        ax = axes[1, 1]
        ax.plot(
            plot_data[var]["times"], plot_data[var]["data"] - 273.15,
        )
        ax.set_title("ERA5")
        ax.set_ylabel("Maximum Temperature (°C)")

        var = "popd"
        ax = axes[2, 0]
        ax.plot(
            plot_data[var]["times"], plot_data[var]["data"],
        )
        ax.set_title("HYDE")
        ax.set_ylabel("Population Density")

        var = "cropland"
        ax = axes[2, 1]
        ax.plot(
            plot_data[var]["times"], plot_data[var]["data"],
        )
        ax.set_title("HYDE")
        ax.set_ylabel("cropland")

        for ax in axes.flatten():
            ax.grid()

        for ax in axes[-1, :]:
            ax.set_xlabel("Date")
        plt.subplots_adjust(wspace=0.22)
    plt.close()
