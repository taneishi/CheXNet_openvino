from openvino.tools.accuracy_checker.annotation_converters.convert import main
from openvino.tools.accuracy_checker.annotation_converters.format_converter import FileBasedAnnotationConverter, ConverterReturn
from openvino.tools.accuracy_checker.representation import MultiLabelRecognitionAnnotation
from openvino.tools.accuracy_checker.config import PathField
import os

from main import CLASS_NAMES

class ChestXRayConverter(FileBasedAnnotationConverter):
    __provider__ = 'chestxray14'
    annotation_types = (MultiLabelRecognitionAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(is_directory=True, description='Path to sample dataset root directory.')
            })
        return parameters

    def configure(self):
        self.data_dir = self.config['data_dir']

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
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
