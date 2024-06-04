import unittest
from unittest.mock import MagicMock
import torch
from mmdet3d.datasets.physion_random_frame_dataset import PhysionRandomFrameDataset  # Import the class containing the method

# class TestIndoorEvalPhysion(unittest.TestCase):
#     def setUp(self):
#         # Mock ground truth annotations (list of indices)
#         self.gt_annos = [{'ann_info': {'gt_labels_3d': torch.tensor([0]),
#                                         'gt_bboxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
#                                         'gt_num': 1}},
#                          {'ann_info': None},
#                          {'ann_info': {'gt_labels_3d': torch.tensor([1]),
#                                         'gt_bboxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
#                                         'gt_num': 1}}]

#         # Mock detection annotations (list of dictionaries)
#         self.dt_annos = [{'labels_3d': torch.tensor([0]),
#                           'boxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
#                           'scores_3d': torch.tensor([0.9])},
#                          {'labels_3d': torch.tensor([1]),
#                           'boxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
#                           'scores_3d': torch.tensor([0.8])}]

#         # Other required inputs
#         self.metric = [0.5, 0.7]
#         self.label2cat = {0: 'class_0', 1: 'class_1'}

#         # Mock other methods or attributes of the class if needed
#         self.mocked_instance = MagicMock(PhysionRandomFrameDataset)  # Create a MagicMock instance of YourClass
#         # Mock other methods or attributes as needed, for example:
#         self.mocked_instance.method_name.return_value = {'object_AP_0.25': 1.0, 'mAP_0.25': 1.0, 'object_rec_0.25': 1.0, 'mAR_0.25': 1.0, 'object_AP_0.50': 1.0, 'mAP_0.50': 1.0, 'object_rec_0.50': 1.0, 'mAR_0.50': 1.0}

#     def test_indoor_eval_physion(self):
#         # Create an instance of YourClass
#         data_root = '/media/kalyanav/Venkat/support_data/'
#         your_class_instance = PhysionRandomFrameDataset(data_root = data_root, ann_file=data_root + 'val_onthefly_data_small.pkl', classes=
#                                                         ['object'], test_mode=True)

#         # Set the mocked instance of YourClass to be returned when calling the method that is called within indoor_eval_physion
#         your_class_instance.method_name = self.mocked_instance.method_name

#         # Call the method using the instance of YourClass
#         ret_dict = your_class_instance.indoor_eval_physion(self.gt_annos, self.dt_annos, self.metric, self.label2cat)
#         import pdb; pdb.set_trace()
#         return ret_dict
#         # Write assertions here to verify the correctness of ret_dict
#         # Example assertions:
#         # self.assertEqual(ret_dict['class_0_AP_0.50'], expected_value)
#         # self.assertEqual(ret_dict['class_1_AP_0.70'], expected_value)
#         # Add more assertions as needed


class TestIndoorEvalPhysion(unittest.TestCase):
    def test_indoor_eval_physion(self):
        # Create an instance of YourClass
        data_root = '/media/kalyanav/Venkat/support_data/'
        your_class_instance = PhysionRandomFrameDataset(data_root = data_root, ann_file=data_root + 'val_onthefly_data_small.pkl', classes=
                                                        ['object'], test_mode=True)

        # Mock input data
        gt_annos = [{'ann_info': {'gt_labels_3d': torch.tensor([0]),
                                   'gt_bboxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
                                   'gt_num': 1}},
                    {'ann_info': None},
                    {'ann_info': {'gt_labels_3d': torch.tensor([1]),
                                   'gt_bboxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
                                   'gt_num': 1}}]

        dt_annos = [{'labels_3d': torch.tensor([0]),
                      'boxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
                      'scores_3d': torch.tensor([0.9])},
                     {'labels_3d': torch.tensor([1]),
                      'boxes_3d': torch.tensor([[0, 0, 0, 1, 1, 1]]),
                      'scores_3d': torch.tensor([0.8])}]

        metric = [0.5, 0.7]
        label2cat = {0: 'class_0', 1: 'class_1'}

        # Call the method using the instance of YourClass
        ret_dict = your_class_instance.indoor_eval_physion(gt_annos, dt_annos, metric, label2cat)

        # Expected ret_dict
        expected_ret_dict = {
            'class_0_AP_0.50': 1.0,
            'class_1_AP_0.50': 1.0,
            'class_0_AP_0.70': 1.0,
            'class_1_AP_0.70': 1.0,
            # Add more expected keys and values as needed
        }

        # Assert that the returned ret_dict matches the expected_ret_dict
        self.assertDictEqual(ret_dict, expected_ret_dict)


if __name__ == '__main__':
    unittest.main()
