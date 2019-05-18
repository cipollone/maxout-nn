'''\
Generic Python methods and classes for a more concise code.
'''


def namespacelike(classN):
  '''\
  Class decorator. This decorator can be used for classes whose instances are
  used as simple namespaces. It changes the dir() method to return just the 
  'public' members of the objects (those not starting with _).
  '''

  def _dir(self):
    l = [k for k in self.__dict__ if not k.startswith('_')]
    l.sort()
    return l

  classN.__dir__ = _dir
  return classN


@namespacelike
class RunContexts:
  '''\
  Create context managers for different runs of a Session. This class can be
  used as:
    cs = RunContexts(sess, train=init_training, test=init_testing, etc...)
    with cs.train:
      # Training
  It generates a context manager for each keyword argument in input. Each 
  context is entered but never left, so that successive `with cs.train' do not
  call the init_training op.
  '''

  def __init__(self, sess, **contexts):
    '''\
    See class description.

    Args:
      sess: tf Session. Must be active when using the contexts.
      key=val: context named 'key' with initialization op 'val'.
    '''

    self._sess = sess
    for name in contexts:
      self.__dict__[name] = self._RunContext(self, name, contexts[name])
    self._current = None

  class _RunContext:
    '''\
    The real context manager class. Internal class: do not use it directly.
    '''

    def __init__(self, allContexts, name, op):
      self.all = allContexts
      self.op = op
      self.name = name

    def __enter__(self):
      if self.all._current != self:
        self.all._sess.run(self.op)
        self.all._current = self

    def __exit__(self, exc_type, exc_value, exc_tb):
      pass
