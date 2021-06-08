import concurrent
import csv
import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict

import requests

from cwine.model.structure import Wine, Country


def _default_row_filter(row):
    type_filter = {'white', 'red', 'rose'}
    reqs = {'winery', 'region', 'year', 'country'}
    size = '0.75'

    for key in reqs:
        if row[key] is None or row[key] == '':
            return False

    if row['type'] != 'simple' or row['has_360'] != 'True':
        return False

    if row['wine_type'] not in type_filter or size not in row['size']:
        return False

    return True


class CsvParser:

    def __init__(self, filter_csv_row=_default_row_filter, csv_feed: str = None):
        self.csv_feed = csv_feed
        if self.csv_feed is None:
            self.csv_feed = os.getenv('CSV_FEED')

        self.filter_csv_row = filter_csv_row
        self.parsed = None

    def parse(self):
        self.parsed = []
        with open(self.csv_feed, newline='\n', encoding='utf-8') as embedding_csv:
            csv_reader = csv.DictReader(embedding_csv)
            for row in csv_reader:
                if self.filter_csv_row(row) is False:
                    continue

                wine = Wine.from_row(row)
                self.parsed.append(wine)


class WineImageSetDownloader:

    def __init__(self, download_path: str, download_limit=-1, json_feed: str = None, s3_image_format: str = None):
        self.download_path = download_path
        self.json_feed = json_feed
        if self.json_feed is None:
            self.json_feed = os.getenv('JSON_FEED')

        self.s3_image_format = s3_image_format
        if self.s3_image_format is None:
            self.s3_image_format = os.getenv('S3_IMAGE_FORMAT')

        self.download_limit = download_limit

        self.failed_skus = []
        self.downloaded = 0

    def read_json_feed(self) -> Dict[str, Country]:
        with open(self.json_feed) as feed:
            return json.load(feed)

    def download_wine(self, wine: Wine):
        wine_downloader = WineImageDownloader(wine)
        directory = os.path.join(self.download_path, wine.sku)
        if os.path.exists(directory):
            files = os.listdir(directory)
            if len(files) >= 24 and any('44' in file for file in files):
                return []
            elif len(files) >= 30:
                return []

        if wine_downloader.perform_async(self.s3_image_format) is True:
            print(f'Downloaded {len(wine_downloader.responses)} images for wine {wine.sku} ({wine.name})')
            for name in wine_downloader.responses:
                res = wine_downloader.responses[name]
                with open(f'{directory}{name}', 'wb') as save:
                    save.write(res)
                    save.close()
        elif wine_downloader.perform_async(self.s3_image_format, ext='jpg') is False:
            print(f"Failed to dump images for {wine.sku} even in jpg")
            self.failed_skus.append(wine.sku)
            return

        self.downloaded += 1

    def download(self):
        structure = self.read_json_feed()
        for country in structure.values():
            for region in country.regions.values():
                for winery in region.wineries.values():
                    for wine in winery.wines.values():
                        if self.download_limit == -1 or self.download_limit > self.downloaded:
                            self.download_wine(wine)
                        else:
                            return


class WineImageDownloader:

    def __init__(self, wine: Wine):
        self.wine = wine
        self.format_types = [
            [44, 88, 132, 176],
            [37, 73, 146, 219, 293],
            [64, 128, 256, 384, 512],
            [85, 171, 341, 512, 683],
        ]
        self.interesting_angles = [0]
        # self.interesting_angles = [0, 1, 5, 6, 7, 11]
        self.responses = dict()

    def perform_async(self, s3_format, ext='png'):
        urls_ft = []
        format_types_len = len(self.format_types)
        for fi in range(format_types_len):
            urls_ft.append([])
            res = self.format_types[fi][-1]
            for angle in self.interesting_angles:
                urls_ft[fi].append(s3_format.format(self.wine.sku, angle, res, ext))

        form = 0
        for urls in urls_ft:
            form = len(urls)
            with ThreadPoolExecutor(max_workers=form) as executor:
                futures = {executor.submit(requests.get, url): url for url in urls}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result: requests.Response = future.result()
                        res_end_index = result.url.index(f'_0_0.{ext}')
                        res_begin_index = result.url.index(f'a_0_') + 4
                        angle_res = result.url[res_begin_index:res_end_index].replace('_', '-')

                        if result.status_code != 200:
                            break

                        self.responses[f'{angle_res}.{ext}'] = result.content
                    except TimeoutError:
                        print("Timeout Error")
            if len(self.responses) > 0:
                break

        response_length = len(self.responses)
        return form >= response_length > 0


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    parser = CsvParser()
    parser.parse()

    dl_helper = WineImageSetDownloader(download_path='../../images/')
    for parsed_wine in parser.parsed:
        dl_helper.download_wine(parsed_wine)
