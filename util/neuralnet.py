from torch import nn

class QNetwork(nn.Module):
	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()  
		self.conv1 = nn.Conv2d(5, 32, kernel_size=8, stride=4)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
		linear_input_size = convw * convh * 64
		self.linear = nn.Linear(linear_input_size, 512)
		self.head = nn.Linear(512, outputs)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = F.relu(self.linear(x.view(x.size(0), -1)))
		return self.head(x)