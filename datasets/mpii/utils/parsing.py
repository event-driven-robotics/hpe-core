import cv2


class MPIIIterator:

    def __init__(self, data_annotations, images_folder):

        self.data = data_annotations['RELEASE']

        # get training indices, i.e. indices of images with pose annotations
        # images without pose annotations are used for evaluation on mpii's servers
        self.training_indices = [ind for ind, is_train in enumerate(self.data['img_train']) if is_train == 1]

        self.images_folder = images_folder

        self.curr_ind = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.training_indices)

    def __next__(self):

        if self.curr_ind == -1:
            raise StopIteration

        # get annotation
        ind = self.training_indices[self.curr_ind]
        self.__update_current_index()

        annotation = self.data['annolist'][ind]

        img = self._get_image(annotation)

        # get pose annotations, i.e. list of dictionaries containing bounding box, keypoints, scale and object position
        pose_ann = self._get_pose_annotation(annotation)

        return img, pose_ann, annotation['image']['name']

    def _get_image(self, annotation):
        image_path = self.images_folder / 'images' / annotation['image']['name']
        return cv2.imread(str(image_path.resolve()))

    def _get_pose_annotation(self, annotation):
        if isinstance(annotation['annorect'], list):
            return annotation['annorect']
        elif isinstance(annotation['annorect'], dict):
            return [annotation['annorect']]

    def __update_current_index(self):

        if self.curr_ind >= self.__len__():
            self.curr_ind = -1
        else:
            self.curr_ind += 1
