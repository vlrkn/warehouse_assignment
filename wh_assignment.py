"""Assign new riders to warehouse taking their preferences into account

Usage:
   wh_assignment.py [options]
   wh_assignment.py --help

Options:
  --help         Print this help.
  --write        Do not write results to spreasheet.

The following environment variables are read:

  * SPREADSHEET_ID: spreadsheet id for input

"""
import os
import time
import gspread
import pandas as pd

from docopt import docopt
from collections import defaultdict
from ortools.linear_solver import pywraplp
from datetime import datetime, date, timedelta
from gspread_dataframe import set_with_dataframe

class WarehouseAssigner:

    def __init__(self, spreadsheet_id):
        self.today = date.today()
        self.spreadsheet_id = spreadsheet_id
        self.df_forecast = None
        self.df_riders = None
        self.df_distance = None
        self.df_stage_id = None
        self.df_opening = None
        self.df_holiday = None
        self.df_already_assigned = None
        self.date_assigned_output = pd.DataFrame(columns=['ID', 'key', 'value'])
        self.warehouse_assigned_output = pd.DataFrame(columns=['ID', 'key', 'value'])
        self.stage_update_output = pd.DataFrame(columns=['id', 'destination'])
        self.riders_score_log_output = pd.DataFrame(columns=['Rider external ID', 'Rider email', 'Score'])

    def get_spreadsheets(self,
                         riders_sheet_name,
                         forecast_sheet_name,
                         distance_sheet_name,
                         index_stage_id_sheet_name,
                         holiday_sheet_name,
                         opening_sheet_name,
                         warehouse_assigned_sheet_name,
                         date_assigned_sheet_name
                         ):

        # Auth to google
        gc = gspread.oauth()

        # Open spreadsheet
        sh = gc.open_by_key(self.spreadsheet_id)

        # Load worksheet data
        riders_worksheet = sh.worksheet(riders_sheet_name)
        forecast_worksheet = sh.worksheet(forecast_sheet_name)
        distance_worksheet = sh.worksheet(distance_sheet_name)
        stage_id_worksheet = sh.worksheet(index_stage_id_sheet_name)
        holiday_worksheet = sh.worksheet(holiday_sheet_name)
        opening_worksheet = sh.worksheet(opening_sheet_name)

        warehouse_assigned_worksheet = sh.worksheet(warehouse_assigned_sheet_name)
        date_assigned_worksheet = sh.worksheet(date_assigned_sheet_name)

        riders_data = riders_worksheet.get_all_values()
        headers = riders_data.pop(0)
        self.df_riders = pd.DataFrame(riders_data, columns=headers)

        forecast_data = forecast_worksheet.get_all_values()
        headers = forecast_data.pop(0)
        self.df_forecast = pd.DataFrame(forecast_data, columns=headers)

        distance_data = distance_worksheet.get_all_values()
        headers = distance_data.pop(0)
        self.df_distance = pd.DataFrame(distance_data, columns=headers)

        stage_id_data = stage_id_worksheet.get_all_values()
        headers = stage_id_data.pop(0)
        self.df_stage_id = pd.DataFrame(stage_id_data, columns=headers)

        holiday_data = holiday_worksheet.get_all_values()
        headers = holiday_data.pop(0)
        self.df_holiday = pd.DataFrame(holiday_data, columns=headers)

        opening_data = opening_worksheet.get_all_values()
        headers = opening_data.pop(0)
        self.df_opening = pd.DataFrame(opening_data, columns=headers)

        warehouse_assigned_data = warehouse_assigned_worksheet.get_all_values()
        headers = warehouse_assigned_data.pop(0)
        df_warehouse_assigned = pd.DataFrame(warehouse_assigned_data, columns=headers).loc[:, ['ID', 'value']]
        df_warehouse_assigned.rename(columns={
            "ID": "External",
            "value": "start_date"
        },
            inplace=True)

        date_assigned_data = date_assigned_worksheet.get_all_values()
        headers = date_assigned_data.pop(0)
        df_date_assigned = pd.DataFrame(date_assigned_data, columns=headers).loc[:, ['ID', 'value']]
        df_date_assigned.rename(columns={
            "ID": "External",
            "value": "Warehouse"
        },
            inplace=True)

        self.df_already_assigned = pd.merge(df_warehouse_assigned, df_date_assigned, how="inner", on=["External"])

    def validate_inputs(self):
        # Check for duplicate riders in rider sheet
        duplicated = self.df_riders[['External']].duplicated()
        if duplicated.any():
            raise NameError('Duplicated riders: {}',
                            self.df_riders.loc[duplicated, ['External']])

        # Check for duplicate in forecast sheet
        duplicated = self.df_forecast[['City', 'Warehouse', 'Week']].duplicated()
        if duplicated.any():
            raise NameError('City, warehouse, week with multiple forecast: {}',
                            self.df_forecast.loc[duplicated, ['City', 'Warehouse', 'Week']])

        # Check for duplicate in opening sheet
        duplicated = self.df_opening[['City', 'Warehouse']].duplicated()
        if duplicated.any():
            raise NameError('City, warehouse with multi opening dates: {}',
                            self.df_opening.loc[duplicated, ['City', 'Warehouse']])

        diff = set(list(zip(self.df_opening.City, self.df_opening.Warehouse))) - set(
            list(zip(self.df_forecast.City, self.df_forecast.Warehouse)))
        if len(diff) > 0:
            raise NameError('Couples (City, Warehouse) with opening date missing in forecast: {}'.format(diff))

    def preprocess_forecast(self):

        # Init assigned columns
        self.df_forecast['Assigned'].replace('', 0, inplace=True)
        self.df_forecast['Need'].replace('', 0, inplace=True)

        # Force type
        self.df_forecast = self.df_forecast.astype({'Warehouse': str,
                                                    'City': str,
                                                    'Week': int,
                                                    # 'Minijob only': str,
                                                    'Need': int,
                                                    'Assigned': int})
        # Remove week from the past
        # self.df_forecast = self.df_forecast.loc[self.df_forecast.Week >= self.today.isocalendar()[1]]

        # self.df_forecast['Minijob only'] = self.df_forecast['Minijob only'].apply(lambda x: 1 if x == 'yes' else 0)

        self.df_forecast['Left'] = self.df_forecast['Need'] - self.df_forecast['Assigned']

        # Get assigned date for each Week
        weekdays = dict()
        for week in set(self.df_forecast.Week):
            # Get weekdays
            startdate = time.asctime(time.strptime('2021 %d 1' % week, '%Y %W %w'))
            startdate = datetime.strptime(startdate, '%a %b %d %H:%M:%S %Y')
            dates = [startdate.strftime('%Y-%m-%d')]
            for i in range(1, 7):
                day = startdate + timedelta(days=i)
                dates.append(day)
            weekdays[week] = dates

        # Max date of the week is Saturday
        self.df_forecast['max_date_to_assign'] = pd.to_datetime(self.df_forecast.apply(
            lambda row: weekdays[row['Week']][5], axis=1))

        # Merge opening date
        # self.df_opening['Opening date'] = pd.to_datetime(self.df_opening['Opening date'])
        self.df_forecast = pd.merge(self.df_forecast, self.df_opening, how="left", on=["City", "Warehouse"])

        # Min date of the week is Monday or opening date for warehouse opening during the week
        self.df_forecast.loc[:, 'min_date_to_assign'] = pd.to_datetime(self.df_forecast.apply(
            lambda row: max(weekdays[row['Week']][0], row['Opening date']) if pd.notnull(row['Opening date']) else
            weekdays[row['Week']][0], axis=1))

        self.df_forecast['forecast_id'] = self.df_forecast['Week'].astype(str) + self.df_forecast['Warehouse']

        # Set index on warehouse name
        self.df_forecast.set_index('forecast_id', inplace=True)
        self.df_forecast.sort_index(inplace=True)

    def get_next_working_day(self, date, city, office=True):
        """
        Get the date if it is working day or the next working date, for office or warehouse
        Closed date -> Sunday or holiday
        For office, also Saturday
        For riders, also on the 14th and 15th (we can't make a rider start on those dates)
        """
        # Saturday -> Monday
        if office and date.weekday() == 5:
            return self.get_next_working_day(date + timedelta(days=2), city, office)
        # Sunday -> Monday
        elif date.weekday() == 6:
            return self.get_next_working_day(date + timedelta(days=1), city, office)
        # 14th or 15th of the month -> 16th
        elif ~office and date.day in (14, 15):
            return self.get_next_working_day(date + timedelta(days=16 - date.day), city, office)
        # holiday -> d+1
        elif date in self.df_holiday.loc[self.df_holiday.City == city, 'Public holiday'].tolist():
            return self.get_next_working_day(date + timedelta(days=1), city, office)
        else:
            return date

    def preprocess_riders(self):
        # Filter out riders without info
        self.df_riders.dropna(how='all', inplace=True)

        # Filter out riders from other cities than Berlin
        self.df_riders = self.df_riders.loc[(self.df_riders['Contract data check'] == 'TRUE')]

        # Extract city from position
        self.df_riders['City'] = self.df_riders['Position'].str.replace(' - Bike Crew Member', '')

        self.df_riders.loc[:, 'Warehouse location preference'] = self.df_riders.apply(
            lambda x: ', '.join([
                pref.strip() for pref in x['Warehouse location preference'].split(',') if
                pref.strip() in self.df_forecast.loc[self.df_forecast.City == x['City'], "Warehouse"].tolist()
            ]), axis=1
        )

        # 2 office open days before assignment possible
        earliest_starting_date_from_now = self.get_next_working_day(self.today + timedelta(days=1),
                                                                    "Berlin",
                                                                    office=True)
        earliest_starting_date_from_now = self.get_next_working_day(earliest_starting_date_from_now,
                                                                    "Berlin",
                                                                    office=False)
        # Fill null value with earliest starting date
        self.df_riders['Earliest start date'] = self.df_riders['Earliest start date'].fillna(
            value=earliest_starting_date_from_now)
        # Set to d+2 or d+4 if before earliest starting date
        self.df_riders.loc[:, 'Earliest start date'] = pd.to_datetime(self.df_riders['Earliest start date']).apply(
            lambda d: d if d >= earliest_starting_date_from_now else earliest_starting_date_from_now)

        self.df_holiday.loc[:, 'Public holiday'] = pd.to_datetime(self.df_holiday['Public holiday'])

        # Get valid start date for riders (no 14th, 15th, holiday or sunday)
        self.df_riders.loc[:, 'Earliest start date'] = self.df_riders.apply(
            lambda row: self.get_next_working_day(
                row['Earliest start date'] + timedelta(days=1),
                row['City'],
                office=False
            ),
            axis=1)

        # Count number of preferences and add a column for this feature
        self.df_riders['nr_preferences'] = self.df_riders[
            'Warehouse location preference'].apply(lambda x: len(x.split(',')))

        # Set index on External / rider id
        self.df_riders.set_index('External', inplace=True)

    def preprocess_distance(self):
        # drop warehouses couple without distance assigned
        self.df_distance.dropna(subset=['Distance in min'], inplace=True)

        # Format warehouse name without special character
        # self.df_distance['WH1'] = self.df_distance['WH1'].str.replace(r' \|.*', '').str.upper()
        # self.df_distance['WH2'] = self.df_distance['WH2'].str.replace(r' \|.*', '').str.upper()

        # Set distance in min type as float
        self.df_distance.loc[:, 'Distance in min'] = self.df_distance['Distance in min'].astype(float)

    def preprocess_stage_id(self):
        city_map = {
            'Dsseldorf': 'Düsseldorf',
            'Kln': 'Cologne',
            'Mnich': 'Munich',
            'Nrnberg': 'Nürnberg'
        }

        self.df_stage_id.loc[:, 'City'] = self.df_stage_id['City'].apply(
            lambda x: city_map[x] if x in city_map.keys() else x)

    def get_inputs(self,
                   riders_sheet_name,
                   forecast_sheet_name,
                   distance_sheet_name,
                   index_stage_id_sheet_name,
                   holiday_sheet_name,
                   opening_sheet_name,
                   warehouse_assigned_sheet_name,
                   date_assigned_sheet_name):

        self.get_spreadsheets(riders_sheet_name,
                              forecast_sheet_name,
                              distance_sheet_name,
                              index_stage_id_sheet_name,
                              holiday_sheet_name,
                              opening_sheet_name,
                              warehouse_assigned_sheet_name,
                              date_assigned_sheet_name)

        self.validate_inputs()

        self.preprocess_forecast()
        self.preprocess_riders()
        self.preprocess_distance()
        self.preprocess_stage_id()

        missing_cities_in_stage_id = set(self.df_forecast.City.unique()) - set(self.df_stage_id.City.tolist())

        if missing_cities_in_stage_id:
            print("City missing in stage id sheet: ", missing_cities_in_stage_id)

        warehouses_not_single_per_city = self.df_forecast.groupby(["City", "Week"]).filter(lambda x: len(x) > 1)[
            'Warehouse']
        warehouses_missing_in_distance = (
                    set(warehouses_not_single_per_city) - set(self.df_distance.WH1.tolist())).union(
            set(warehouses_not_single_per_city) - set(self.df_distance.WH2.tolist()))

        if warehouses_missing_in_distance:
            print("Warehouse missing in distance sheet: ", warehouses_missing_in_distance)

        wrong_warehouses_in_distance = (set(self.df_distance.WH1.tolist())).union(
            set(self.df_distance.WH2.tolist())) - set(warehouses_not_single_per_city)

        if wrong_warehouses_in_distance:
            print("Warehouse misspelled in distance sheet: ", wrong_warehouses_in_distance)

    def rider_pref_score(self, row, df_forecast):
        d = defaultdict(int)

        if row['Warehouse location preference']:
            warehouse_prefs = [pref.strip() for pref in row['Warehouse location preference'].split(',')]
        else:
            warehouse_prefs = []

        # Iterate over the warehouses
        for index, w in df_forecast.iterrows():

            # If warehouse is not in the pref warehouses, assign average distance to
            if warehouse_prefs and not w['Warehouse'] in warehouse_prefs:
                distance = self.df_distance.loc[
                    (self.df_distance.WH1 == w['Warehouse']) &
                    (self.df_distance.WH2.isin(warehouse_prefs)), 'Distance in min']
                if distance.empty:
                    distance = self.df_distance.loc[
                        (self.df_distance.WH2 == w['Warehouse']) &
                        (self.df_distance.WH1.isin(warehouse_prefs)), 'Distance in min']
                d[index] = 100 / distance.mean()
            else:
                d[index] = 100
        return pd.Series(d)

    def compute_pref_score(self, forecast, riders):

        df_preferences = riders.apply(lambda x: self.rider_pref_score(x, forecast), axis=1)
        df_preferences = df_preferences.reindex(sorted(df_preferences.columns), axis=1)

        return df_preferences

    def solve_riders_assignment(self, forecast, riders, preferences):

        solver = pywraplp.Solver('SolveAssignmentProblemMIP',
                                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        cost = preferences.values.tolist()

        limit_w = forecast['Left']

        num_riders = len(cost)
        num_warehouses = len(cost[0])

        x = {}

        for i in range(num_riders):
            for j in range(num_warehouses):
                x[i, j] = solver.BoolVar('x[%i,%i]' % (i, j))

        solver.Maximize(solver.Sum([cost[i][j] * x[i, j] for i in range(num_riders)
                                    for j in range(num_warehouses)]))

        # Each rider is assigned to at most 1 task.
        for i in range(num_riders):
            solver.Add(solver.Sum([x[i, j] for j in range(num_warehouses)]) <= 1)

        # Each warehouse has at most left needs riders assigned
        for j in range(num_warehouses):
            solver.Add(solver.Sum([x[i, j] for i in range(num_riders)]) <= limit_w[j])

        # Each warehouse has at least some riders assigned
        for j in range(num_warehouses):
            lower_limit = (limit_w[j] * min(num_riders / sum(limit_w), 1)) // 1
            solver.Add(solver.Sum([x[i, j] for i in range(num_riders)]) >= lower_limit)

        sol = solver.Solve()

        res = []
        for i in range(num_riders):
            for j in range(num_warehouses):
                if x[i, j].solution_value() > 0:
                    start_date = max(forecast.loc[[preferences.columns[j]], 'min_date_to_assign'].tolist()[0],
                                     riders.loc[[preferences.index[i]], 'Earliest start date'].tolist()[0])
                    res.append({
                        'rider_id': preferences.index[i],
                        'forecast_id': preferences.columns[j],
                        'warehouse': forecast.loc[[preferences.columns[j]], 'Warehouse'].tolist()[0],
                        'start_date': start_date,
                        'cost': cost[i][j],
                        'email': riders.loc[[preferences.index[i]], 'Email'].tolist()[0]
                    })
        df_assigned = pd.DataFrame(res)

        return df_assigned

    def update_output(self, df_assigned):

        batch_date_assigned = df_assigned[['rider_id', 'start_date']]
        batch_date_assigned.loc[:, 'key'] = 'earliest_start_date'
        batch_date_assigned.rename(columns={'rider_id': 'ID',
                                            'start_date': 'value'}, inplace=True)
        batch_date_assigned = batch_date_assigned[['ID', 'key', 'value']]
        batch_date_assigned.loc[:, 'value'] = batch_date_assigned.value.astype(str)
        self.date_assigned_output = self.date_assigned_output.append(batch_date_assigned, ignore_index=True)

        batch_warehouse_assigned = df_assigned[['rider_id', 'warehouse']]
        batch_warehouse_assigned.loc[:, 'key'] = 'warehouse_location_preference'
        batch_warehouse_assigned.rename(columns={'rider_id': 'ID',
                                                 'warehouse': 'value'}, inplace=True)
        self.warehouse_assigned_output = self.warehouse_assigned_output.append(batch_warehouse_assigned)

        batch_stage_update = df_assigned[['rider_id', 'destination']]
        batch_stage_update.rename(columns={'rider_id': 'id'}, inplace=True)
        self.stage_update_output = self.stage_update_output.append(batch_stage_update,
                                                                   ignore_index=True)

        batch_riders_score_log = df_assigned[['rider_id', 'email', 'cost']]
        batch_riders_score_log.rename(columns={'rider_id': 'Rider external ID',
                                               'email': 'Rider email',
                                               'cost': 'Score'}, inplace=True)
        self.riders_score_log_output = self.riders_score_log_output.append(batch_riders_score_log,
                                                                           ignore_index=True)

        assignment_counts = df_assigned.forecast_id.value_counts()
        assignment_counts = assignment_counts.to_frame().rename(columns={'forecast_id': 'Assigned'})
        self.df_forecast.Assigned = self.df_forecast.apply(
            lambda x: (x['Assigned'] if x.name not in assignment_counts.index else x['Assigned'] + assignment_counts.at[
                x.name, 'Assigned']), axis=1)

    def assign_single_warehouse_in_city(self, forecast, riders):
        res = []
        forecast_id = forecast.first_valid_index()
        left_need = forecast.at[forecast_id, 'Left']
        warehouse = forecast.at[forecast_id, 'Warehouse']
        for index, rider in riders.head(left_need).iterrows():
            start_date = max(forecast.at[forecast_id, 'min_date_to_assign'],
                             rider['Earliest start date'])
            res.append({'rider_id': index,
                        'forecast_id': forecast_id,
                        'warehouse': warehouse,
                        'start_date': start_date,
                        'cost': 100,
                        'email': riders['Email']})

        df_assigned = pd.DataFrame(res)
        return df_assigned

    def assign_riders(self):
        for week in self.df_forecast.Week.unique():
            for city in self.df_forecast.City.unique():
                # Select forecast for city and current week
                forecast = self.df_forecast.loc[(self.df_forecast.Week == week) &
                                                (self.df_forecast.City == city) &
                                                (self.df_forecast.max_date_to_assign >= pd.to_datetime(
                                                    self.today + timedelta(days=2))) &
                                                (self.df_forecast.Left > 0),
                           :]
                if not forecast.loc[forecast.Left > 0, :].empty:

                    week_end_date = forecast['max_date_to_assign'].tolist()[0]
                    riders = self.df_riders.loc[(~self.df_riders.index.isin(self.df_already_assigned.External)) &
                                                (~self.df_riders.index.isin(self.warehouse_assigned_output.ID)) &
                                                (self.df_riders['Earliest start date'] <= week_end_date) &
                                                (self.df_riders.City == city), :]

                    if not riders.empty:
                        # Handle cities with single warehouse
                        if len(forecast.Warehouse.tolist()) == 1:
                            df_assigned = self.assign_single_warehouse_in_city(forecast, riders)
                        # Handle generic case
                        else:
                            preferences = self.compute_pref_score(forecast, riders)

                            df_assigned = self.solve_riders_assignment(forecast, riders, preferences)

                        if not df_assigned.empty:
                            # Get city destination ID
                            df_assigned['destination'] = self.df_stage_id.loc[
                                self.df_stage_id.City == city, 'Stage ID contract sending'].tolist()[0]
                            print(
                                "W" + str(week) + " in " + city + ": " + str(df_assigned.shape[0]) + " riders assigned")
                            # print(df_assigned.warehouse.value_counts().to_dict())
                            self.update_output(df_assigned)
                        else:
                            print("No assignment possible for w" + str(week) + " in " + city)
                    else:
                        print("No riders for w" + str(week) + " in " + city)
                else:
                    print("No forecast for w" + str(week) + " in " + city)

    def write_output(self):

        if not self.warehouse_assigned_output.empty:
            gc = gspread.oauth()

            sh = gc.open_by_key(self.spreadsheet_id)

            forecast_worksheet = sh.worksheet('Input/output 1 Forecast')
            date_assigned_worksheet = sh.worksheet('Output 2 Date assigned')
            wh_assigned_worksheet = sh.worksheet('Output 3 WH assigned')
            stage_update_worksheet = sh.worksheet('Output 4 Stage update')
            riders_score_worksheet = sh.worksheet('Output 5 Riders score log')

            # CLEAR date_assigned_output SHEET CONTENT
            range_of_cells = date_assigned_worksheet.range('A2:C1000')  # -> Select the range you want to clear
            for cell in range_of_cells:
                cell.value = ''
            date_assigned_worksheet.update_cells(range_of_cells)
            # APPEND date_assigned_output to SHEET CONTENT
            set_with_dataframe(date_assigned_worksheet, self.date_assigned_output)

            # CLEAR wh_assigned_worksheet SHEET CONTENT
            range_of_cells = wh_assigned_worksheet.range('A2:C1000')  # -> Select the range you want to clear
            for cell in range_of_cells:
                cell.value = ''
            wh_assigned_worksheet.update_cells(range_of_cells)
            # APPEND wh_assigned_worksheet to SHEET CONTENT
            set_with_dataframe(wh_assigned_worksheet, self.warehouse_assigned_output)

            # CLEAR stage_update_worksheet SHEET CONTENT
            range_of_cells = stage_update_worksheet.range('A2:C1000')  # -> Select the range you want to clear
            for cell in range_of_cells:
                cell.value = ''
            stage_update_worksheet.update_cells(range_of_cells)
            # APPEND stage_update_worksheet to SHEET CONTENT
            set_with_dataframe(stage_update_worksheet, self.stage_update_output)

            # APPEND riders_score_worksheet TO SHEET CONTENT
            set_with_dataframe(riders_score_worksheet, self.riders_score_log_output)

            # CLEAR forecast_worksheet SHEET CONTENT
            range_of_cells = forecast_worksheet.range('A2:C1000')  # -> Select the range you want to clear
            for cell in range_of_cells:
                cell.value = ''
            forecast_worksheet.update_cells(range_of_cells)
            # APPEND forecast_worksheet TO SHEET CONTENT
            set_with_dataframe(forecast_worksheet,
                               self.df_forecast[['City', 'Warehouse', 'Week', 'Need', 'Assigned']].sort_values(
                                   by=['Week', 'City', 'Warehouse']))


def main(args):

    write = args['--write']

    riders_sheet_name = 'Input 2 Riders preference'
    forecast_sheet_name = 'Input/output 1 Forecast'
    distance_sheet_name = 'Input 3 Distance WH'
    index_stage_id_sheet_name = 'Index Stage ID'
    holiday_sheet_name = 'Input 5 Public holidays'
    opening_sheet_name = 'Input 4 Openings dates'
    date_assigned_sheet_name = 'Output 2 Date assigned'
    warehouse_assigned_sheet_name = 'Output 3 WH assigned'
    spreadsheet_id = os.environ['SPREADSHEET_ID']

    warehouse_assigner = WarehouseAssigner(spreadsheet_id)

    warehouse_assigner.get_inputs(
        riders_sheet_name=riders_sheet_name,
        forecast_sheet_name=forecast_sheet_name,
        distance_sheet_name=distance_sheet_name,
        index_stage_id_sheet_name=index_stage_id_sheet_name,
        holiday_sheet_name=holiday_sheet_name,
        opening_sheet_name=opening_sheet_name,
        warehouse_assigned_sheet_name=warehouse_assigned_sheet_name,
        date_assigned_sheet_name=date_assigned_sheet_name
    )

    warehouse_assigner.assign_riders()

    if write:
        warehouse_assigner.write_output()

if __name__ == '__main__':
    # command line arguments
    args = docopt(__doc__)
    main(args)