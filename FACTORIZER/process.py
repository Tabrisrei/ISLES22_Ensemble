import json
import os
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from predict import predict

DEFAULT_INPUT_PATH = Path("../input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("../output_teams/factorizer/")


# todo change with your team-name
class Factorizer:
    def __init__(
        self,
        input_path: Path = DEFAULT_INPUT_PATH,
        output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH,
    ):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path("~/factorizer/test/")
            self._output_path = Path("~/factorizer/test/output/")
            self._algorithm_output_path = self._output_path

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image', 'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = (
            input_data["dwi_image"],
            input_data["adc_image"],
            input_data["flair_image"],
        )


        prediction = predict(dwi_image, adc_image, flair_image)

        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = (
            input_data["dwi_image"].GetOrigin(),
            input_data["dwi_image"].GetSpacing(),
            input_data["dwi_image"].GetDirection(),
        )

        # Segment images.
        prediction = self.predict(input_data)  # function you need to update!

        # Build the itk object.
        output_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
        output_image.SetOrigin(origin), output_image.SetSpacing(
            spacing
        ), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = os.path.join(self._algorithm_output_path, 'lesion_msk.nii.gz')
        sitk.WriteImage(output_image, str(output_image_path))


    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(
            slug="dwi-brain-mri", filetype="image"
        )
        adc_image_path = self.get_file_path(
            slug="adc-brain-mri", filetype="image"
        )
        flair_image_path = self.get_file_path(
            slug="flair-brain-mri", filetype="image"
        )


        input_data = {
            "dwi_image": sitk.ReadImage(str(dwi_image_path)),
            "adc_image": sitk.ReadImage(str(adc_image_path)),
            "flair_image": sitk.ReadImage(str(flair_image_path))
                    }

        # Set input information.
        input_filename = str(dwi_image_path).split("/")[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype="image"):
        """ Gets the path for each MR image/json file."""

        if filetype == "image":
            file_list = list(
                (self._input_path / "images" / slug).glob("*.nii.gz")
            )
        elif filetype == "json":
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print("Loading error")
        else:
            return file_list[0]

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    Factorizer().process()
