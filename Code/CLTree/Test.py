
import numpy  as np
class Data(object):
	"""docstring for Data"""
	def __init__(self,data,**kwargs):
		"""
		Parameters
		----------
		data:   numpy array data and last column is lael
		X: input  data
		Y: target data
   
		Returns
		-------
		score : the final score
		"""
		super(Data, self).__init__()
		if not isinstance (data,np.ndarray):
			self.data=np.array(data)
		self.X=self.data[:,:-1]
		self.Y=self.data[:,-1]

	@property
	def n_columns(self):
		nc=self.data.shape[1]
		if nc>0:
			return int(nc)
		else:
			raise AttributeError('Data set is empty')
	@property
	def column_names(self):
		if 'column_names' not in self.__dict__.keys():
			names=['index '+str(i) for i in xrange(0,self.n_columns)]
			mydic={}
			for i, name in enumerate(names):
				mydic[i]=str(name)  
			self.__dict__['column_names']=mydic
		return self.__dict__['column_names']
		
	@column_names.setter
	def column_names(self,names=None):

		n_names=len(names)
		if n_names != self.n_columns:
			raise AttributeError('Column Name Length Conflict')
		mydic={}
		for i, name in enumerate(names):
			mydic[i]=str(name)  
		self.__dict__['column_names']=mydic
	@property
	def ranges(self):
		ranges={}
		for j in xrange(0,len(self.X[0])):
			ranges[j]=(min(self.X[:,j]), max(self.X[:,j]))
		return ranges
			







