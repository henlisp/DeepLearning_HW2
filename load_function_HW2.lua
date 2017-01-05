require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local opt = lapp[[
    -r,--learningRate  (default 0.1)        learning rate
    -m,--momentum      (default 0.9)        momentum
	-b,--batch         (default 128)        number of batches
	
]]


-----### Load the data

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')
local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)


	--normalizing our data

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    --print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    --print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values
for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


do -- data augmentation module
	local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

	function BatchFlip:__init()
		parent.__init(self)
		self.train = true
	end

		-- Flip Image : 	
	local function hflip(x)
		return torch.random(0,1) == 1 and data or image.hflip(x)
	end

	function BatchFlip:updateOutput(input)
		if self.train then
			local bs = input:size(1)
			local flip_mask = torch.randperm(bs):le(bs/2)
			for i=1,input:size(1) do
				--image=require('image')
				if flip_mask[i] == 1  then image.hflip(input[i]) end 
			end
		end
		self.output:set(input:cuda())
		return self.output
	end

	
  -- function BatchFlip:updateOutput(input)
    -- if self.train then
      -- local permutation = torch.randperm(input:size(1))
      -- for i=1,input:size(1) do
        -- if 0 == permutation[i] % 3  then f(input[i]) end -- need to define f
		-- if 1 == permutation[i] % 3  then g(input[i]) end -- need to define g
      -- end
    -- end
    -- self.output:set(input)
    -- return self.output
  -- end
 -- end

	  
	  --random cropped
	  
	local function randomcrop(im , pad, randomcrop_type)
	   if randomcrop_type == 'reflection' then
		  --Each feature map of a given input is padded with the replication of the input boundary
		  module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
	   elseif randomcrop_type == 'zero' then
		  --Each feature map of a given input is padded with specified number of zeros.
		  --If padding values are negative, then input is cropped.
		  module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
	   end
		
	   local padded = module:forward(im:float())
	   local x = torch.random(1,pad*2 + 1)
	   local y = torch.random(1,pad*2 + 1)
	   image.save('img2ZeroPadded.jpg', padded)

	   return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
	end

end


------ ### Load a saved model, run it on the test set, and return the average error : 
local filename = "model.t7"
model = torch.load(filename)

local batchSize = opt.batch

---- ### Classification criterion

criterion = nn.ClassNLLCriterion():cuda()

require 'optim'
image = require 'image'

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    
    model:evaluate() -- turn of drop-out
   
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)--:cuda()
        local yt = labels:narrow(1, i, batchSize)--:cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end


	
--- ### return the average error on the test set

	testLoss, testError, confusion = forwardNet(testData, testLabels, false)
	print('Test error: ' .. testError, 'Test Loss: ' .. testLoss)
