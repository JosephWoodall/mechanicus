import pandas


class Movement:
    """This class contains functions that leverage the output of the Inference class and move the servo to the appropriate position"""

    def lookup_tuple_value_from_hash(
        inference_data: pandas.DataFrame = None, hash_value: str = None
    ):
        """
        Look up the move_to_position tuple value using a hash value.

        Args:
            inference_data (pd.DataFrame): DataFrame containing 'prosthetic_cartesian_3d_position' and 'prosthetic_cartesian_3d_position_hash_value' columns
            hash_value (int): The prosthetic_cartesian_3d_position_hash_value value to look up

        Returns:
            tuple: The corresponding move_to_position tuple

        Raises:
            ValueError: If no matching hash value is found
        """
        # Find the matching row
        matching_row = inference_data[
            inference_data["prosthetic_cartesian_3d_position_hash_value"] == hash_value
        ]

        # Check if we found exactly one match
        if len(matching_row) == 0:
            raise ValueError(f"No tuple found for hash value {hash_value}")
        elif len(matching_row) > 1:
            raise ValueError(f"Multiple tuples found for hash value {hash_value}")

        # Return the move_to_position tuple
        return matching_row["move_to_position"].iloc[0]

    def move_prosthetic_to_appropriate_position(prosthetic_cartesian_3d_position):
        # ingest the tuple value found from the prediction and move the prosthetic to its position
        # please note, the tuple value from the prediciton must be extracted using the prosthetic_cartesian_3d_position_hash_value value
        # simply lookup the prosthetic_cartesian_3d_position_hash_value's respective tuple value, found from prosthetic_cartesian_3d_position, and pass it for the function
        move_to_position = prosthetic_cartesian_3d_position
        return int(move_to_position)
