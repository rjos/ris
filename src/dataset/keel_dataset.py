from sklearn.model_selection import StratifiedKFold
from src.dataset.utils import to_numpy, to_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

class KeelAttribute:
    TYPE_REAL, TYPE_INTEGER, TYPE_NOMINAL = range(3)

    def __init__(self, attribute_name, attribute_type, attribute_range, builder):
        self.name = attribute_name
        self.type = attribute_type
        self.range = attribute_range
        self.builder = builder


class KeelDataSet:
    UNKNOWN = '?'

    def __init__(self, relation_name, attributes, data, inputs=None, outputs=None):
        self.name = relation_name
        self.attributes = attributes
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.shape = len(data[0]), len(data)

    def get_data(self, attributes):
        return [self.data[self.attributes.index(a)] for a in attributes]

    def get_data_target(self, inputs=None, outputs=None):
        inputs = inputs or self.inputs
        outputs = outputs or self.outputs

        assert inputs and outputs, 'You should specify inputs and outputs either here or in the initialization.'

        # return self.get_data(inputs), self.get_data(outputs)
        return np.transpose(self.get_data(inputs)), np.concatenate(self.get_data(outputs))
    
    def describe(self):
        lbls = np.array(self.get_data(self.outputs)[0])
        uniques, counts = np.unique(lbls, return_counts=True)
        # unique_classes = set(self.get_data(self.outputs)[0])
        return {'name': self.name, 'instances': self.shape[0], 'attributes': len(self.inputs), 'classes': ('-').join(list(map(str, counts)))}

    def get_header(self):
        header = ('@relation {name}\n').format(name = self.name)
        header += ('\n').join([('@attribute {name} {type} [{min}, {max}]').format(name = attr.name, type = ('integer' if attr.type == KeelAttribute.TYPE_INTEGER else 'real'), min = attr.range[0], max = attr.range[1]) for attr in self.attributes])
        header += '\n'
        header += ('@inputs {inputs}\n').format(inputs = (', ').join([attr.name for attr in self.inputs]))
        header += ('@outputs {outputs}\n').format(outputs = (', ').join([attr.name for attr in self.outputs]))
        header += '@data\n'
        return header
    
    def get_folders(self, num_folders = 10, normalize = False):
        folders = []

        k_fold = StratifiedKFold(n_splits=num_folders, shuffle=True)

        inp, out = self.get_data_target()
        
        # inp = to_numpy(inp)
        # out = to_numpy(out)
        # out = np.concatenate(out)

        minmax = MinMaxScaler()

        for i, (train, test) in enumerate(k_fold.split(inp, out)):
            x_train, x_test = inp[train], inp[test]
            y_train, y_test = out[train], out[test]

            if normalize:
                x_train = minmax.fit_transform(x_train)
                x_test = minmax.fit_transform(x_test)

            data = to_data(x_train, y_train.reshape(y_train.shape[0], 1), axis=1)

            fold_train = KeelDataSet(('{name}-{num}-{f}tra.dat').format(name = self.name, num=num_folders, f = (i+1)), self.attributes, data, self.inputs, self.outputs)

            data = to_data(x_test, y_test.reshape(y_test.shape[0], 1), axis=1)
            fold_test = KeelDataSet(('{name}-{num}-{f}tst.dat').format(name = self.name, num = num_folders, f = (i+1)), self.attributes, data, self.inputs, self.outputs)

            folders.append((fold_train, fold_test))

        return folders
    
    def save(self, path):
        with open(path, 'w') as file:
            file.write(self.get_header())
            data = list(map(list, zip(*self.data)))
            for i, d in enumerate(data):
                data[i] = list(map(lambda x,y: x.builder(y), self.attributes, d))
            data = ('\n').join(map((', ').join, map(lambda x: map(str, x), data)))
            file.write(data)

def __parse_attributes_list(l, lkp):
    ret = []
    for i in l[:-1]:
        # Will remove the trailling coma
        ret.append(lkp[i[:-1]])
    ret.append(lkp[l[-1]])
    return ret


def load_from_file(file):
    is_str = type(file) == str
    handle = open(file) if is_str else file
    try:
        top = handle.readline()

        l = top.split()
        if l[0] != '@relation' or len(l) != 2:
            raise SyntaxError('This is not a valid keel database.')

        relation_name = l[1]

        line = handle.readline().strip()

        attrs = []
        lkp = {}
        while line.startswith('@attribute'):
            l = line.split(maxsplit=3)
            if len(l) == 3:
                spl = re.split(r'(\w+)(\[.*\])', l[2])
                if len(spl) == 4:
                    l[2] = spl[1]
                    l.append(spl[2])

            if len(l) != 4:
                raise NotImplementedError('This is probably a nominal parameter. We don\'t have support for this yet.',
                                          l, file)
            if l[2][0] != '{':
                name = l[1]
                a_type = l[2]
                a_range = l[3]
            else:
                l = line.split(maxsplit=2)
                name = l[1]
                a_type = 'nominal'
                a_range = l[2]

            if a_type == 'real':
                a_type = KeelAttribute.TYPE_REAL
                builder = float
            elif a_type == 'integer':
                a_type = KeelAttribute.TYPE_INTEGER
                builder = float
            elif a_type == 'nominal':
                a_type = KeelAttribute.TYPE_NOMINAL
                builder = str
            else:
                raise SyntaxError('Unknown type.')

            if a_type != KeelAttribute.TYPE_NOMINAL:
                a_min, a_max = a_range[1:-1].split(',')
                # print(file, a_range[1:-1])
                a_range = builder(a_min), builder(a_max)
            else:
                a_range = a_range[1:-1].replace(' ', '').split(',')

            k = KeelAttribute(name, a_type, a_range, builder)
            attrs.append(k)
            lkp[name] = k
            line = handle.readline().strip()

        l = line.split()
        if l[0] != '@inputs':
            raise SyntaxError('Expected @inputs.' + line + str(l))
        inputs = __parse_attributes_list(l[1:], lkp)

        line = handle.readline()
        l = line.split()

        if l[0] != '@outputs':
            raise SyntaxError('Expected @outputs.')

        outputs = __parse_attributes_list(l[1:], lkp)

        line = handle.readline()

        if line != '@data\n':
            raise SyntaxError('Expected @data.')

        data = [[] for _ in range(len(attrs))]
        for data_line in handle:
            if data_line:
                l = data_line.strip().split(',')
                for lst, value, attr in zip(data, l, attrs):
                    v = value
                    v = v if v == KeelDataSet.UNKNOWN else attr.builder(v)
                    lst.append(v)

        return KeelDataSet(relation_name, attrs, data, inputs, outputs)
    finally:
        if is_str:
            handle.close()
