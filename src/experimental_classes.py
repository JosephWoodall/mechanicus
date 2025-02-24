import numpy
import pandas


class HemisphericalGenerator:

    def __init__(self, n_theta: int = 10, n_phi: int = 10, radius=1):
        """Generates points along a flattened cartesian plane from a half-sphere.

        Args:
            n_theta (int, optional): Number of samples along the theta dimension. Defaults to 10.
            n_phi (int, optional): NUmber of samples along the phi dimension. Defaults to 10.
            radius (int, optional): Radius of the half-sphere. Defaults to 1.
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.radius = radius

    def generate_hemisphere_points(self):
        theta = numpy.linspace(0, 2 * numpy.pi, self.n_theta)
        phi = numpy.linspace(0, numpy.pi / 2, self.n_phi)

        theta, phi = numpy.meshgrid(theta, phi)

        x = self.radius * numpy.sin(phi) * numpy.cos(theta)
        y = self.radius * numpy.sin(phi) * numpy.sin(theta)
        z = self.radius * numpy.cos(phi)

        points = numpy.column_stack((x.flatten(), y.flatten(), z.flatten()))
        return points


class CartesianPlanPositionOfMovement:
    def __init__(self, rows: int = 10000, cols: int = 10):
        self.rows = rows
        self.cols = cols

    def generate_offline_eeg_data_with_cartesian_plane_position_of_movement(
        self,
    ) -> pandas.DataFrame:
        """Generates synthetic eeg data which matches to a 3D cartesian plane, with mean of 0 (pointed at zenith) of prosthetic movement.
        The idea is that upon each movement, eeg data corresponds to the angle of which the prosthetic will point. So at 0 (zenith), the
        prosthetic will be directionally parallel to the limb it is upon. In other words, the prosthetic will be straight.
        Ultimately, this is what the data will look like, so it is being generated here. All 3d_position values will be greater than 0 because
        the prosthetic cannot physically invert upon itself.

        The intuition is that I am able to predict what the 3D position of the prosthetic is based on the corresponding eeg data.

        Returns:
            pandas.DataFrame: _description_
        """
        # Generate features matrix with incremental values
        base_features = numpy.arange(self.rows * self.cols).reshape(
            self.rows, self.cols
        )

        # Center and normalize features to mean=0, std=1
        features_mean = base_features.mean()
        features_std = base_features.std(
            ddof=1
        )  # Using ddof=1 for sample standard deviation
        features_normalized = (base_features - features_mean) / features_std

        # Create DataFrame with features
        df = pandas.DataFrame(
            features_normalized, columns=[f"eeg_{i}" for i in range(self.cols)]
        )

        # Generate response values (3-tuples with mean=0, std=1)
        response_base = numpy.random.normal(loc=0, scale=1, size=(self.rows, 3))
        response_mean = response_base.mean(axis=1, keepdims=True)
        response_std = response_base.std(axis=1, keepdims=True)
        response_normalized = (response_base - response_mean) / response_std

        # Ensure all values are >= 0 while preserving mean=0, std=1
        response_positive = numpy.abs(response_normalized)
        response_mean_final = response_positive.mean(axis=1, keepdims=True)
        response_std_final = response_positive.std(axis=1, keepdims=True)
        response_final = (response_positive - response_mean_final) / response_std_final

        # Add response column as tuples
        df["prosthetic_cartesian_3d_position"] = [tuple(row) for row in response_final]
        df["prosthetic_cartesian_3d_position_hash_value"] = df[
            "prosthetic_cartesian_3d_position"
        ].apply(lambda x: abs(hash(",".join(f"{v:.10f}" for v in x))))

        return df


if __name__ == "__main__":

    n = 100
    points = HemisphericalGenerator(n, n, n).generate_hemisphere_points()
    print(points)

    grid = (
        CartesianPlanPositionOfMovement().generate_offline_eeg_data_with_cartesian_plane_position_of_movement()
    )
    print(grid)
