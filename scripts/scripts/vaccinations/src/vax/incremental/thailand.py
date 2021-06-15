import re
import requests
from datetime import datetime
import tempfile

from bs4 import BeautifulSoup
import pandas as pd
from pdfreader import SimplePDFViewer

from vax.utils.incremental import enrich_data, increment, clean_count, merge_with_current_data
from vax.utils.utils import get_soup
from vax.utils.dates import clean_date


class Thailand:

    def __init__(self):
        self.location = "Thailand"
        self.source_url = "https://ddc.moph.go.th/dcd/pagecontent.php?page=643&dept=dcd"
        self.regex = {
            "total_vaccinations": r"ผู้ที่ได้รับวัคซีนสะสม .{1,100} ทั้งหมด[^\d]+([\d,]+) โดส",
            "people_vaccinated": r"ผู้ได้รับวัคซีนเข็มที่ 1 .{1,3}นวน[^\d]+([\d,]+) ?ร.{1,3}ย",
            "people_fully_vaccinated": (
                r"นวนผู้ได้รับวัคซีนครบต.{1,2}มเกณฑ์ \(ได้รับวัคซีน 2 เข็ม\) .{1,3}นวน[^\d]+([\d,]+)"
            ),
            "date": r"\s?ข้อมูล ณ วันที่ (\d{1,2}) (.*) (\d{4})",
        }

    def read(self, last_update: str) -> pd.DataFrame:
        yearly_report_page = get_soup(self.source_url)
        # Get Newest Month Report Page
        monthly_report_link = yearly_report_page.find("div", class_="col-lg-12", id="content-detail").find("a")["href"]
        monthly_report_page = get_soup(monthly_report_link)
        # Get links
        df = self._parse_data(monthly_report_page, last_update)
        return df

    def _parse_data(self, monthly_report_page: BeautifulSoup, last_update: str):
        elems = monthly_report_page.find("div", class_="col-lg-12", id="content-detail")
        links = [link["href"].replace(" ", "%20") for link in elems.find_all("a")]
        records = []
        for link in links:
            # print("-------------------")
            # print(link)
            record, stop = self._parse_data_date(link, last_update)
            if stop:
                break
            records.append(record)
        return pd.DataFrame(records)

    def _parse_data_date(self, link: str, last_update: str) -> dict:
        # Get text from PDF
        raw_text = self._text_from_pdf(link)
        text = self._substitute_special_chars(raw_text)
        # Get date
        date_str = self._parse_date(text)
        if date_str < last_update:
            return None, True
        data = {
            "total_vaccinations": self._parse_metric(text, "total_vaccinations"),
            "people_vaccinated": self._parse_metric(text, "people_vaccinated"),
            "people_fully_vaccinated": self._parse_metric(text, "people_fully_vaccinated"),
            "date": date_str,
            "source_url": link.replace(" ", "%20"),
        }
        return data, False

    def _text_from_pdf(self, pdf_link: str):
        with tempfile.NamedTemporaryFile() as tf:
            with open(tf.name, mode="wb") as f:
                f.write(requests.get(pdf_link).content)

            with open(tf.name, mode="rb") as f:
                viewer = SimplePDFViewer(f)
                viewer.render()
                raw_text = "".join(viewer.canvas.strings)
        return raw_text

    def _substitute_special_chars(self, raw_text: str):
        """Correct Thai Special Character Error."""
        special_char_replace = {
            '\uf701': u'\u0e34',
            '\uf702': u'\u0e35',
            '\uf703': u'\u0e36',
            '\uf704': u'\u0e37',
            '\uf705': u'\u0e48',
            '\uf706': u'\u0e49',
            '\uf70a': u'\u0e48',
            '\uf70b': u'\u0e49',
            '\uf70e': u'\u0e4c',
            '\uf710': u'\u0e31',
            '\uf712': u'\u0e47',
            '\uf713': u'\u0e48',
            '\uf714': u'\u0e49',
        }
        special_char_replace = dict((re.escape(k), v) for k, v in special_char_replace.items())
        pattern = re.compile("|".join(special_char_replace.keys()))
        text = pattern.sub(lambda m: special_char_replace[re.escape(m.group(0))], raw_text)
        return text

    def _parse_metric(self, text: str, metric_name: str):
        if metric_name not in self.regex:
            raise ValueError(f"Metric {metric_name} not a valid metric!")
        total_vaccinations = re.search(self.regex[metric_name], text).group(1)
        return clean_count(total_vaccinations)

    def _parse_date(self, text: str):
        thai_date_replace = {
            # Months
            "มกราคม": 1,
            "กุมภาพันธ์": 2,
            "มีนาคม": 3,
            "เมษายน": 4,
            "พฤษภาคม": 5,
            "พฤษภำคม": 5,
            "มิถุนายน": 6,
            "มิถุนำยน": 6,
            "กรกฎาคม": 7,
            "สิงหาคม": 8,
            "กันยายน": 9,
            "ตุลาคม": 10,
            "พฤศจิกายน": 11,
            "ธันวาคม": 12,
        }
        date_raw = re.search(self.regex["date"], text)
        day = clean_count(date_raw.group(1))
        month = thai_date_replace[date_raw.group(2)]
        year = clean_count(date_raw.group(3)) - 543
        return clean_date(datetime(year, month, day))

    def pipe_location(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(location=self.location)

    def pipe_vaccine(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(vaccine="Oxford/AstraZeneca, Sinovac")

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df
            .pipe(self.pipe_location)
            .pipe(self.pipe_vaccine)
        )

    def to_csv(self, paths):
        output_file = paths.tmp_vax_out(self.location)
        last_update = pd.read_csv(output_file).date.max()
        df = self.read(last_update)
        if not df.empty:
            df = df.pipe(self.pipeline)
            df = merge_with_current_data(df, output_file)
            df.to_csv(output_file, index=False)


def main(paths):
    Thailand().to_csv(paths)


if __name__ == "__main__":
    main()
