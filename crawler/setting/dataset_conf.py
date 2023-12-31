from datetime import datetime

from dateutil.relativedelta import relativedelta

from configuration import Configuration


class DatasetConfiguration(Configuration):

    def _format_date(self, date_str):
        return datetime.strptime(date_str, '%Y-%m-%d')

    def _calculate_step(self, step):
        step = int(step)
        if self.step_unit == 'day':
            return relativedelta(days=step)
        elif self.step_unit == 'month':
            return relativedelta(months=step)
        else:
            return relativedelta(years=step)

    def _init_properties(self):
        return [
            ['name', '', str],
            ['base_api_url', '', str],
            ['start_date', '2022-07-02', self._format_date],
            ['end_date', datetime.now().strftime('%Y-%m-%d'), self._format_date],
            ['step_unit', 'day', str],
            ['step', 1, self._calculate_step],
            ['path', '/content/drive/MyDrive/New_crawler/', str],
            ['sleep', 0.2, float],
            ['start_page','',str],
            ['end_page','',str],
            ['user','',str],
            ['pw','',str]
        ]
