import re
import os
import warnings



def get_checkpoint(resume_dir):
    """Get checkpoint

    Arguments:
        resume_dir {str} -- path to the dir containing the checkpoint

    Raises:
        IOError -- No checkpoint has been found

    Returns:
        str -- path to the checkpoint
    """

    models_list = [
        f for f in os.listdir(resume_dir) if f.endswith(".pth.tar")
    ]
    models_list.sort()

    if not models_list:
        raise IOError(
            'Directory {} does not contain any model'.format(resume_dir))

    if len(models_list) > 1:
        model_name = models_list[-2]
    else:
        model_name = models_list[-1]

    return os.path.join(resume_dir, model_name)


def ensure_dir(path):
    """Make sure directory exists, otherwise
    create it.

    Arguments:
        path {str} -- path to the directory

    Returns:
        str -- path to the directory
    """

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_filename_from_path(path):
    """Get name of a file given the absolute or
    relative path

    Arguments:
        path {str} -- path to the file

    Returns:
        str -- file name without format
    """

    assert isinstance(path, str)
    file_with_format = os.path.split(path)[-1]
    file_format = re.findall(r'\.[a-zA-Z]+', file_with_format)[-1]
    file_name = file_with_format.replace(file_format, '')

    return file_name, file_format


def file_exists(path):
    """Check if file exists

    Arguments:
        path {str} -- path to the file

    Returns:
        bool -- True if file exists
    """

    assert isinstance(path, str)
    return os.path.exists(path)





def abs_path(path):
    """Get absolute path of a relative one

    Arguments:
        path {str} -- relative path

    Raises:
        NameError -- String is empty

    Returns:
        str -- absolute path
    """

    assert isinstance(path, str)
    if path:
        return os.path.expanduser(path)

    raise NameError('Path is empty...')


def get_dir(path):
    """Get directory name from absolute or
    relative path

    Arguments:
        path {str} -- path to directory

    Returns:
        str -- directory name
    """

    assert isinstance(path, str)
    if '.' in path[-4:]:
        name = path.split('/')[-1]
        dir_name = path.replace('/{}'.format(name), '')
        return dir_name

    return path

def get_parent(path):
    assert isinstance(path, str)
    sep='/'
    split_name= path.split(sep)
    print("st",split_name)
    name=sep.join(split_name[:-1])
    return name



def remove_files(paths):
    """Delete files

    Arguments:
        paths {list} -- list of paths
    """

    assert isinstance(paths, list)
    for path in paths:
        os.remove(path)





def get_sub_dirs(path):
    """Get sub-directories contained in a specified directory

    Arguments:
        path {str} -- path to directory

    Returns:
        name -- list of names of the sub-directories
        paths -- list of paths of the sub-directories
    """
    try:
        dirs = os.walk(path).next()[1] #root, dirs, files
    except AttributeError:
        try:
            dirs = next(os.walk(path))[1]
        except StopIteration:
            dirs = []
    dirs.sort()
    dir_paths = [os.path.join(path, i) for i in dirs]
    return dirs, dir_paths


def get_files(path, file_format):
    """Get file paths of files contained in
    a given directory according to the format

    Arguments:
        path {str} -- path to the directory containing the files
        file_format {list | str} -- list or single format #jpg

    Returns:
        paths, names -- lists of paths and names
    """
    #if
    if isinstance(file_format, str):
        file_format = [file_format]
    else:
        assert isinstance(file_format, list)
    # get sub-directories files
    _, sub_dirs = get_sub_dirs(path)
    if sub_dirs:
        warnings.warn("the function to get files is accessing subdirectories path:"+path)
        list_paths = []
        list_names = []
        for sub_dir in sub_dirs:
            paths, names = get_files(sub_dir, file_format)
            list_paths.extend(paths)
            list_names.extend(names)
        return list_paths, list_names
    # get current files
    file_names = []
    file_paths = []
    for f_format in file_format:
        if f_format != '*':
            files = [f for f in os.listdir(path)
                     if re.match(r'.*\.{}'.format(f_format), f)]
        else:
            files = os.listdir(path)
        files.sort()

        file_names.extend([f.replace('.{}'.format(f_format), '')
                           for f in files])
        file_paths.extend([os.path.join(path, f) for f in files])

    return file_paths, file_names
