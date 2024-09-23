import SimpleITK
import numpy as np
import json
import os
from pathlib import Path
import torch
from torch.cuda.amp import autocast
from monai import transforms, data
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch
import sys
sys.path.append('.')


DEFAULT_INPUT_PATH = Path("../input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("../output_teams/nvauto/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("../output_teams/nvauto/results.json")


# todo change with your team-name
class ThresholdModel():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path('./test')
            self._output_path = Path('./test/output')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image_path, adc_image_path, flair_image_path = input_data['dwi_image_path'],\
                                                            input_data['adc_image_path'],\
                                                            input_data['flair_image_path']


        ##################
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True


        load_keys=['image']
   
        val_transform = []
        val_transform.append(transforms.LoadImaged(keys=load_keys))
        val_transform.append(transforms.EnsureChannelFirstd(keys=load_keys))
        val_transform.append(transforms.CastToTyped(keys=['image'], dtype=np.float32))
        val_transform.append(transforms.EnsureTyped(keys=load_keys, data_type='tensor'))
        
        val_transform.append(transforms.Spacingd(keys=['image'], pixdim=[1,1,1], mode=['bilinear']))
        val_transform.append(transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True))

        val_transform = transforms.Compose(val_transform)

        validation_files = [{"image": [adc_image_path, dwi_image_path]}]
        #dirname = os.path.dirname(__file__)
        dirname = os.path.dirname(os.getcwd())
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, sampler=None)


        checkpoints = [ os.path.join(dirname, 'weights/NVAUTO/ts/model0.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model1.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model2.ts'),

                        os.path.join(dirname, 'weights/NVAUTO/ts/model3.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model4.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model5.ts'),

                        os.path.join(dirname, 'weights/NVAUTO/ts/model6.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model7.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model8.ts'),

                        os.path.join(dirname, 'weights/NVAUTO/ts/model9.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model10.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model11.ts'),

                        os.path.join(dirname, 'weights/NVAUTO/ts/model12.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model13.ts'),
                        os.path.join(dirname, 'weights/NVAUTO/ts/model14.ts'),

                        ]

        model_inferer = SlidingWindowInferer(roi_size=[192, 192, 128], overlap=0.625, mode='gaussian', cache_roi_weight_map=True, sw_batch_size=2)

        with torch.no_grad():
            for idx, batch_data in enumerate(val_loader):
                image = batch_data['image'].cuda(0)

                all_probs=[]
                for checkpoint in checkpoints:
                    #print('Inference with', checkpoint)
              
                    model = torch.jit.load(checkpoint)
                    model.cuda(0)
                    model.eval()


                    with autocast(enabled=True):
                        logits = model_inferer(inputs=image, network=model)  # another inferer (e.g. sliding window)

                    probs = torch.softmax(logits.float(), dim=1)

                    batch_data["pred"] = probs
                    inverter = transforms.Invertd(keys="pred", transform=val_transform, orig_keys="image", meta_keys="pred_meta_dict", nearest_interp=False, to_tensor=True)
                    probs = [inverter(x)["pred"] for x in decollate_batch(batch_data)] #invert resampling if any
                    probs = torch.stack(probs, dim=0)
                    # print('inverted resampling', logits.shape)

                    all_probs.append(probs.cpu())

                probs = sum(all_probs)/len(all_probs) #mean
                labels = torch.argmax(probs, dim=1).cpu().numpy().astype(np.int8)

                prediction = labels[0].copy()

        prediction = prediction.transpose((2, 1, 0))

        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):

        dwi_image_data = SimpleITK.GetArrayFromImage(input_data['dwi_image'])


        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['dwi_image'].GetOrigin(),\
                                     input_data['dwi_image'].GetSpacing(),\
                                     input_data['dwi_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction.astype(np.uint8))
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = os.path.join(self._algorithm_output_path, 'lesion_msk.nii.gz')
        SimpleITK.WriteImage(output_image, str(output_image_path))


    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug='dwi-brain-mri', filetype='image')
        adc_image_path = self.get_file_path(slug='adc-brain-mri', filetype='image')
        flair_image_path = self.get_file_path(slug='flair-brain-mri', filetype='image')


        input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 'dwi_image_path': str(dwi_image_path), 
                      'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 'adc_image_path': str(adc_image_path),
                      'flair_image': SimpleITK.ReadImage(str(flair_image_path)), 'flair_image_path': str(flair_image_path)}


        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            file_list = list((self._input_path / "images" / slug).glob("*.nii.gz"))
        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    ThresholdModel().process()
