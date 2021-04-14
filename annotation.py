from accuracy_checker.annotation_converters.convert import main
from accuracy_checker.annotation_converters.format_converter import FileBasedAnnotationConverter, ConverterReturn
from accuracy_checker.representation import MultiLabelRecognitionAnnotation
from accuracy_checker.config import PathField
from accuracy_checker.utils import get_path
import os

from main import CLASS_NAMES

class ChestXRayConverter(FileBasedAnnotationConverter):
    __provider__ = 'chest_xray'
    annotation_types = (MultiLabelRecognitionAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(is_directory=True, description='Path to sample dataset root directory.')
            })
        return parameters

    def configure(self):
        '''
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        '''
        self.data_dir = self.config['data_dir']

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        '''
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata (if provided)
        '''
        # read and convert annotation
        image_list_file = os.path.join('labels', 'val_list.txt')
        
        annotations= []
        with open(image_list_file, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = [int(i) for i in items[1:]]
                annotations.append(MultiLabelRecognitionAnnotation(image_name, label))
        return ConverterReturn(annotations, self.generate_meta(CLASS_NAMES), None)

    @staticmethod
    def generate_meta(labels):
        return {'label_map': {value:key for value,key in enumerate(labels)}}

if __name__ == '__main__':
    main()
