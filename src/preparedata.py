import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os


class readGPDData:
    def __init__(self, dataprepinargs):
        self.dataparam = dataprepinargs
        self.wd = dataprepinargs["workingdir"]
        self.nyears = 31

    def readfiles(self):
        covardir = self.dataparam["covars"]
        inventorydir = self.dataparam["inventory"]
        slopeunitdir = self.dataparam["slopeunits"]
        self.covars = pd.read_csv(self.wd + covardir).dropna()
        self.df_inventory = gpd.read_file(self.wd + inventorydir).dropna()
        self.df_slopeunit = gpd.read_file(self.wd + slopeunitdir).dropna()
        self.df_inventory = self.df_inventory.to_crs(epsg=self.dataparam["epsg"])
        self.df_slopeunit = self.df_slopeunit.to_crs(epsg=self.dataparam["epsg"])
        litho = pd.get_dummies(self.covars.Litho)
        litho.columns = litho.columns.str.replace("_", "")
        self.covars[litho.columns] = litho.to_numpy()

    def get_target(self, year):
        df_slopeunit_sub = self.df_slopeunit
        monsoon_year = year
        inventory_subset = self.df_inventory[
            self.df_inventory["Monsoon_Ye"] == str(monsoon_year)
        ]
        if inventory_subset.empty:
            return None
        sjoined = gpd.overlay(inventory_subset, df_slopeunit_sub, how="intersection")
        sjoined["area_new"] = sjoined.area
        sjoined = sjoined.groupby(["cat"])["area_new"].agg(["sum", "count"])
        sjoined = sjoined.rename(columns={"sum": "area_inventory"})
        df_slopeunit_sub["area_slopeunit"] = df_slopeunit_sub.area
        df_slopeunit_sub = df_slopeunit_sub.join(sjoined, on="cat")
        df_slopeunit_sub = df_slopeunit_sub.fillna(0)
        df_slopeunit_sub["landslide"] = df_slopeunit_sub.area_inventory > 0.0
        df_slopeunit_sub["landslide"] = df_slopeunit_sub["landslide"].astype(int)
        df_slopeunit_sub["area_density"] = 100 * (
            df_slopeunit_sub.area_inventory / df_slopeunit_sub.area_slopeunit
        )
        df_slopeunit_sub = df_slopeunit_sub[
            ["cat", "area_density", "landslide", "count"]
        ]
        return df_slopeunit_sub

    def preparedata(self):
        self.readfiles()
        constant_covars = self.covars[self.dataparam["constcols"]]
        for i in tqdm(range(self.nyears)):
            ndvi_mean_col = f"{str(i)}_NdMx_mean"
            ndvi_stdv_col = f"{str(i)}_NdMx_stdDev"
            prec_mean_col = f"b{str(i+1)}_PMx"
            prec_stdv_col = f"b{str(i+1)}_PMe"
            prec_maxi_col = f"b{str(i+1)}_PSt"
            covars_subset = self.covars[
                [
                    "cat",
                    ndvi_mean_col,
                    ndvi_stdv_col,
                    prec_mean_col,
                    prec_stdv_col,
                    prec_maxi_col,
                ]
            ]
            covars_subset = covars_subset.rename(
                columns={
                    ndvi_mean_col: "ndviMe",
                    ndvi_stdv_col: "ndviSt",
                    prec_mean_col: "precMe",
                    prec_stdv_col: "precSt",
                    prec_maxi_col: "precMx",
                }
            )
            covar_data = pd.merge(
                constant_covars,
                covars_subset,
                left_on="cat",
                right_on="cat",
                left_index=False,
            )

            # create target variables now
            monsoon_year = self.dataparam["startyear"] + i

            df_target = self.get_target(monsoon_year)
            if df_target is None:
                continue
            alldata = pd.merge(
                covar_data, df_target, left_on="cat", right_on="cat", left_index=False
            )
            if i == 0:
                clean_covar = alldata
            else:
                clean_covar = pd.concat([clean_covar, alldata])
        clean_covar = clean_covar.dropna()
        Xtrain = clean_covar[self.dataparam["variables"]].to_numpy()
        clean_covar["area_density"]=clean_covar["area_density"]
        Ytrain = clean_covar[["landslide", "area_density", "count"]].to_numpy()
        if self.dataparam["removezeros"]:
            idx = np.where(Ytrain > 0)[0]
            Xtrain = Xtrain[idx]
            Ytrain = Ytrain[idx]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            Xtrain, Ytrain, test_size=self.dataparam["testsize"], random_state=420
        )
        self.covars = None
        self.df_inventory = None
        self.df_slopeunit = None

    def preparedatainference(self, return_period=5):
        self.readfiles()
        design_rainfall = pd.read_csv(self.wd + self.dataparam["design_rainfall"])
        constant_covars = self.covars[self.dataparam["constcols"]]
        # calculate the mean NDVI
        for i in tqdm(range(self.nyears)):
            ndvi_mean_col = f"{str(i)}_NdMx_mean"
            ndvi_stdv_col = f"{str(i)}_NdMx_stdDev"
            # prec_mean_col=f'b{str(i+1)}_PMx'
            # prec_stdv_col=f'b{str(i+1)}_PMe'
            # prec_maxi_col=f'b{str(i+1)}_PSt'
            covars_subset = self.covars[["cat", ndvi_mean_col, ndvi_stdv_col]]
            covars_subset = covars_subset.rename(
                columns={ndvi_mean_col: "ndviMe", ndvi_stdv_col: "ndviSt"}
            )
            covar_data = covars_subset

            if i == 0:
                ndvis = covar_data
            else:
                ndvis = pd.concat([ndvis, covar_data])
        ndvis = ndvis.groupby(["cat"]).mean().reset_index()

        alldata = pd.merge(
            constant_covars, ndvis, left_on="cat", right_on="cat", left_index=False
        )
        # now add rainfall data
        rainfall_subset = design_rainfall[
            [
                "cat",
                f"st_design_{return_period}",
                f"me_design_{return_period}",
                f"mx_design_{return_period}",
            ]
        ]
        rainfall_subset = rainfall_subset.rename(
            columns={
                f"me_design_{return_period}": "precMe",
                f"st_design_{return_period}": "precSt",
                f"mx_design_{return_period}": "precMx",
            }
        )

        clean_covar = pd.merge(
            alldata, rainfall_subset, left_on="cat", right_on="cat", left_index=False
        )

        clean_covar = clean_covar.dropna()
        self.Xinference = clean_covar[self.dataparam["variables"]].to_numpy()
        self.InferenceID = clean_covar["cat"].to_numpy()
        self.covars = None
        self.df_inventory = None
        self.df_slopeunit = None

    def preparedataclimate(self, rp=5, model="CanESM2", scenario="rcp45"):
        self.readfiles()
        design_rainfall = pd.read_csv(
            self.wd + self.dataparam["climateprojections_rainfall"]
        )
        constant_covars = self.covars[self.dataparam["constcols"]]
        # calculate the mean NDVI
        for i in tqdm(range(self.nyears)):
            ndvi_mean_col = f"{str(i)}_NdMx_mean"
            ndvi_stdv_col = f"{str(i)}_NdMx_stdDev"
            # prec_mean_col=f'b{str(i+1)}_PMx'
            # prec_stdv_col=f'b{str(i+1)}_PMe'
            # prec_maxi_col=f'b{str(i+1)}_PSt'
            covars_subset = self.covars[["cat", ndvi_mean_col, ndvi_stdv_col]]
            covars_subset = covars_subset.rename(
                columns={ndvi_mean_col: "ndviMe", ndvi_stdv_col: "ndviSt"}
            )
            covar_data = covars_subset

            if i == 0:
                ndvis = covar_data
            else:
                ndvis = pd.concat([ndvis, covar_data])
        ndvis = ndvis.groupby(["cat"]).mean().reset_index()

        alldata = pd.merge(
            constant_covars, ndvis, left_on="cat", right_on="cat", left_index=False
        )
        # now add rainfall data
        rainfall_subset = design_rainfall[
            [
                "cat",
                f"mx_{rp}_{scenario}_{model}",
                f"me_{rp}_{scenario}_{model}",
                f"st_{rp}_{scenario}_{model}",
            ]
        ]
        rainfall_subset = rainfall_subset.rename(
            columns={
                f"me_{rp}_{scenario}_{model}": "precMe",
                f"st_{rp}_{scenario}_{model}": "precSt",
                f"mx_{rp}_{scenario}_{model}": "precMx",
            }
        )

        clean_covar = pd.merge(
            alldata, rainfall_subset, left_on="cat", right_on="cat", left_index=False
        )

        clean_covar = clean_covar.dropna()
        self.Xinference = clean_covar[self.dataparam["variables"]].to_numpy()
        self.InferenceID = clean_covar["cat"].to_numpy()
        self.covars = None
        self.df_inventory = None
        self.df_slopeunit = None
