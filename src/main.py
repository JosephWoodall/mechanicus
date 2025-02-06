import pandas
import numpy
import glob
import os
import random


class DataCollector:

    def generate_offline_eeg_data(rows: int = 64, cols: int = 9759) -> pandas.DataFrame:
        """Creates a pandas DataFrame similar to offline EEG data with random float values between -1 and 1.

        Args:
            rows (int, optional): number of rows for the ouput. Defaults to 64.
            cols (int, optional): number of columns for the output. Defaults to 9759.

        Returns:
            pandas.DataFrame: pandas DataFrame similar to offline EEG data with random float values between -1 and 1.
        """

        data = numpy.random.uniform(low=-1, high=1, size=(rows, cols))

        column_names = [f"{i+1}" for i in range(cols)]

        return_data = pandas.DataFrame(data, columns=column_names)

        return_data["activity_type"] = random.choice(
            ["baseline_eyes_open", "baseline_eyes_closed"]
        )

        return return_data

    def collect_offline_eeg_data() -> pandas.DataFrame:
        """This method collects data from https://www.physionet.org/content/eegmmidb/1.0.0/S001/#files-panel. A summary of the data is shown below:

        Abstract
        This data set consists of over 1500 one- and two-minute EEG recordings, obtained from 109 volunteers, as described below.

        Experimental Protocol
        Subjects performed different motor/imagery tasks while 64-channel EEG were recorded using the BCI2000 system (http://www.bci2000.org). Each subject performed 14 experimental runs: two one-minute baseline runs (one with eyes open, one with eyes closed), and three two-minute runs of each of the four following tasks:

        - A target appears on either the left or the right side of the screen. The subject opens and closes the corresponding fist until the target disappears. Then the subject relaxes.
        - A target appears on either the left or the right side of the screen. The subject imagines opening and closing the corresponding fist until the target disappears. Then the subject relaxes.
        - A target appears on either the top or the bottom of the screen. The subject opens and closes either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears. Then the subject relaxes.
        A-  target appears on either the top or the bottom of the screen. The subject imagines opening and closing either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears. Then the subject relaxes.
        In summary, the experimental runs were:

        1. Baseline, eyes open
        2. Baseline, eyes closed
        3. Task 1 (open and close left or right fist)
        4. Task 2 (imagine opening and closing left or right fist)
        5. Task 3 (open and close both fists or both feet)
        6. Task 4 (imagine opening and closing both fists or both feet)
        7. Task 1
        8. Task 2
        9. Task 3
        10. Task 4
        11. Task 1
        12. Task 2
        13. Task 3
        14. Task 4
        The data are provided here in EDF+ format (containing 64 EEG signals, each sampled at 160 samples per second, and an annotation channel). For use with PhysioToolkit software, rdedfann generated a separate PhysioBank-compatible annotation file (with the suffix .event) for each recording. The .event files and the annotation channels in the corresponding .edf files contain identical data.

        Each annotation includes one of three codes (T0, T1, or T2):

        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
        the left fist (in runs 3, 4, 7, 8, 11, and 12)
        both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
        the right fist (in runs 3, 4, 7, 8, 11, and 12)
        both feet (in runs 5, 6, 9, 10, 13, and 14)
        In the BCI2000-format versions of these files, which may be available from the contributors of this data set, these annotations are encoded as values of 0, 1, or 2 in the TargetCode state variable.

        Montage
        The EEGs were recorded from 64 electrodes as per the international 10-10 system (excluding electrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10), as shown in this PDF figure. The numbers below each electrode name indicate the order in which they appear in the records; note that signals in the records are numbered from 0 to 63, while the numbers in the figure range from 1 to 64.

        Acknowledgments
        This data set was created and contributed to PhysioBank by Gerwin Schalk (schalk at wadsworth dot org) and his colleagues at the BCI R&D Program, Wadsworth Center, New York State Department of Health, Albany, NY. W.A. Sarnacki collected the data. Aditya Joshi compiled the dataset and prepared the documentation. D.J. McFarland and J.R. Wolpaw were responsible for experimental design and project oversight, respectively. This work was supported by grants from NIH/NIBIB ((EB006356 (GS) and EB00856 (JRW and GS)).



        Returns:
            pandas.DataFrame: instance of the data collected offline.
        """
        import mne

        data_dir = "src/data/eeg-motor-movementimagery-dataset-1.0.0/files"

        filenames = glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True)

        dataframes = []

        for i in filenames:
            activity_type_in_i = i[i.find("R") + 1 : i.find(".edf")]

            eeg_data = mne.io.read_raw_edf(f"{i}").get_data()

            eeg_data = pandas.DataFrame(eeg_data)

            if activity_type_in_i == "01":
                eeg_data["activity_type"] = "baseline_eyes_open"
            elif activity_type_in_i == "02":
                eeg_data["activity_type"] = "baseline_eyes_closed"
            elif activity_type_in_i == "03":
                eeg_data["activity_type"] = "task_1_open_and_close_left_or_right_fist"
            elif activity_type_in_i == "04":
                eeg_data["activity_type"] = (
                    "task_2_imagine_opening_and_closing_left_or_right_fist"
                )
            elif activity_type_in_i == "05":
                eeg_data["activity_type"] = (
                    "task_3_open_and_close_both_fists_or_both_feet"
                )
            elif activity_type_in_i == "06":
                eeg_data["activity_type"] = (
                    "task_4_imagine_opening_and_closing_both_fists_or_feet"
                )
            elif activity_type_in_i == "07":
                eeg_data["activity_type"] = "task_1"
            elif activity_type_in_i == "08":
                eeg_data["activity_type"] = "task_2"
            elif activity_type_in_i == "09":
                eeg_data["activity_type"] = "task_3"
            elif activity_type_in_i == "10":
                eeg_data["activity_type"] = "task_4"
            elif activity_type_in_i == "11":
                eeg_data["activity_type"] = "task_1"
            elif activity_type_in_i == "12":
                eeg_data["activity_type"] = "task_2"
            elif activity_type_in_i == "13":
                eeg_data["activity_type"] = "task_3"
            elif activity_type_in_i == "14":
                eeg_data["activity_type"] = "task_4"
            else:
                eeg_data["activity_type"] = "none_found_from_source_file"

            eeg_data["subject_id"] = i
            dataframes.append(eeg_data)

        output_eeg_data = pandas.concat(dataframes, ignore_index=True)

        return output_eeg_data


production_data = DataCollector.collect_offline_eeg_data()
print(production_data)
