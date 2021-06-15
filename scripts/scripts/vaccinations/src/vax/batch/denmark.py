import requests

import pandas as pd


class Denmark:

    def __init__(self):
        self.location = "Denmark"
        self.source_url_ref = "https://covid19.ssi.dk/overvagningsdata/vaccinationstilslutning"

    def read(self, source: str) -> str:
        data = requests.get(source).json()
        return pd.DataFrame.from_records(elem["attributes"] for elem in data["features"])

    def pipe_rename_columns(self, df: pd.DataFrame, colname: str) -> pd.DataFrame:
        df.columns = ("date", colname)
        return df

    def pipe_format_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(date=pd.to_datetime(df.date, unit="ms"))

    def pipe_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df
            .groupby("date")
            .sum()
            .sort_values("date")
            .cumsum()
            .reset_index()
        )

    def pipe_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            people_vaccinated=df.people_vaccinated.ffill(),
            people_fully_vaccinated=df.people_fully_vaccinated.ffill()
        )
        return df.assign(
            total_vaccinations=df.people_vaccinated.fillna(0) + df.people_fully_vaccinated.fillna(0)
        )

    def pipe_vaccine(self, df: pd.DataFrame) -> pd.DataFrame:
        def _enrich_vaccine(date: str) -> str:
            if date >= "2021-04-14":
                return "Moderna, Pfizer/BioNTech"
            if date >= "2021-02-08":
                return "Moderna, Oxford/AstraZeneca, Pfizer/BioNTech"
            if date >= "2021-01-13":
                return "Moderna, Pfizer/BioNTech"
            return "Pfizer/BioNTech"
        return df.assign(vaccine=df.date.astype(str).apply(_enrich_vaccine))

    def pipe_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            location=self.location,
            source_url=self.source_url_ref
        )

    def pipeline(self, df: pd.DataFrame, colname: str) -> pd.DataFrame:
        return (
            df
            .pipe(self.pipe_rename_columns, colname)
            .pipe(self.pipe_format_date)
            .pipe(self.pipe_aggregate)
        )

    def pipe_filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.date >= "2020-12-01"]
        return df

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df
            .pipe(self.pipe_metrics)
            .pipe(self.pipe_vaccine)
            .pipe(self.pipe_metadata)
            .pipe(self.pipe_filter_rows)
        )

    def to_csv(self, paths):
        source_dose1 = (
            "https://services5.arcgis.com/Hx7l9qUpAnKPyvNz/ArcGIS/rest/services/Vaccine_REG_linelist_gdb/"
            "FeatureServer/19/query?where=1%3D1&objectIds=&time=&resultType=none&"
            "outFields=first_vaccinedate%2Cantal_foerste_vacc&"
            "returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&"
            "cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&"
            "resultRecordCount=&sqlFormat=none&f=pjson&token="
        )
        source_dose2 = (
            "https://services5.arcgis.com/Hx7l9qUpAnKPyvNz/ArcGIS/rest/services/Vaccine_REG_linelist_gdb/"
            "FeatureServer/20/query?where=1%3D1&objectIds=&time=&resultType=none&"
            "outFields=second_vaccinedate%2Cantal_faerdig_vacc&"
            "returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&"
            "cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&"
            "resultRecordCount=&sqlFormat=none&f=pjson&token="
        )
        destination = paths.tmp_vax_out("Denmark")

        dose1 = self.read(source_dose1).pipe(self.pipeline, colname="people_vaccinated")
        dose2 = self.read(source_dose2).pipe(self.pipeline, colname="people_fully_vaccinated")

        (
            pd.merge(dose1, dose2, how="outer", on="date")
            .pipe(self.post_process)
            .to_csv(destination, index=False)
        )


def main(paths):
    Denmark().to_csv(paths)


if __name__ == "__main__":
    main()
