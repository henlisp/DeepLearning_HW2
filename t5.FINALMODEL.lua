--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 

mkdir t5
cd t5/
git clone https://github.com/soumith/cifar.torch.git
cd cifar.torch/
th Cifar10BinToTensor.lua

]]


require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local opt = lapp[[
    -s,--save          (default 1)          save the model
    -l,--load          (default 0)          load the model
    -o,--optimization  (default "adam")     optimization type
    -n,--network       (default "n2")       reload pretrained network
    -e,--epoch         (default 20)         number of epochs
    -b,--batch         (default 128)        number of batches
    -r,--learningRate  (default 0.1)        learning rate
    -m,--momentum      (default 0.9)        momentum
    -f,--filename      (default "none")     file name to save result
    -d,--dropout       (default 0.5)        dropout probability
    -a,--augmentation  (default 1)          data augmentation
]]


local filename = "model.t7"

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

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


--  ****************************************************************
--  Define our neural network
--  ****************************************************************


function addAndPrint(m, layer)
    local ww, dE_dww = m:getParameters()
    local prev_num = ww:nElement()
    m:add(layer)
    ww, dE_dww = m:getParameters()
    local next_num = ww:nElement()
    local num = next_num - prev_num
	
    --local img = trainData[100]:cuda()
    --local output = m:forward(img)
    --local outSize = output:size()
	
    --print('Number of parameters: ', num, ' layer: ', layer)
    print(num, layer)
end


do -- data augmentation module
	local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

	function BatchFlip:__init()
		parent.__init(self)
		self.train = true
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



local model = nn.Sequential()

-- Create a layer with normalization and ReLU
local function Block(...)
  local arg = {...}
  addAndPrint(model, nn.SpatialConvolution(...))
  --model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

if opt.network == 'n1' then
	model:add(cudnn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
	model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
	model:add(cudnn.ReLU(true))                          -- ReLU activation function
	model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
	model:add(cudnn.SpatialConvolution(32, 64, 3, 3))
	model:add(cudnn.SpatialMaxPooling(2,2,2,2))
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(64))
	model:add(cudnn.SpatialConvolution(64, 32, 3, 3))
	model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
	model:add(nn.Linear(32*4*4, 256))             -- fully connected layer (matrix multiplication between input and weights)
	model:add(cudnn.ReLU(true))
	model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
	model:add(nn.Linear(256, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits)
	model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati
elseif opt.network == 'n2' then
	print('Params', 'Layer')
	addAndPrint(model, nn.BatchFlip())
	addAndPrint(model, nn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
	addAndPrint(model, nn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
	addAndPrint(model, nn.ReLU(true))                          -- ReLU activation function
	addAndPrint(model, nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
	addAndPrint(model, nn.SpatialConvolution(32, 64, 3, 3))
	addAndPrint(model, nn.SpatialMaxPooling(2,2,2,2))
	addAndPrint(model, nn.ReLU(true))
	addAndPrint(model, nn.SpatialBatchNormalization(64))
	addAndPrint(model, nn.SpatialConvolution(64, 24, 3, 3))
	addAndPrint(model, nn.View(24*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
	addAndPrint(model, nn.Linear(24*4*4, 32))             -- fully connected layer (matrix multiplication between input and weights)
	addAndPrint(model, nn.ReLU(true))
	if opt.dropout ~= 0 then
		addAndPrint(model, nn.Dropout(opt.dropout))       	--Dropout layer with p=0.5 (or with defined number)
	end
	addAndPrint(model, nn.Linear(32, #classes))            	-- 10 is the number of outputs of the network (in this case, 10 digits)
	addAndPrint(model, nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati
elseif opt.network == 'n3' then
	print('Params', 'Layer')
        addAndPrint(model, nn.BatchFlip())
 	Block(3,64,5,5,1,1,2,2)
	Block(64,32,1,1)
	Block(32,32,1,1)
	model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        if opt.dropout ~= 0 then
 		model:add(nn.Dropout(opt.dropout))
	end
	Block(32,32,5,5,1,1,2,2)
	Block(32,64,1,1)
	Block(64,32,1,1)
	model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
        if opt.dropout ~= 0 then
 		model:add(nn.Dropout(opt.dropout))
        end
	Block(32,32,3,3,1,1,1,1)
	Block(32,48,1,1)
	Block(48,10,1,1)
	model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
	model:add(nn.View(10))

	for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
		v.weight:normal(0,0.05)
		v.bias:zero()
	end
    addAndPrint(model, nn.LogSoftMax())

elseif opt.network == 'n4' then
        print('Params', 'Layer')

        addAndPrint(model, nn.BatchFlip())
        addAndPrint(model, nn.SpatialConvolution(3, 8, 5, 5))   -- 3x32x32 -> 8x28x28
        addAndPrint(model, nn.ReLU(true))               
        addAndPrint(model, nn.SpatialBatchNormalization(8))

        addAndPrint(model, nn.SpatialConvolution(8, 16, 3, 3))  -- 8x28x28 -> 16x26x26
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(16))

        addAndPrint(model, nn.SpatialConvolution(16, 16, 1, 1))  -- 16x26x26 -> 16x26x26
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(16))

        addAndPrint(model, nn.SpatialConvolution(16, 32, 3, 3)) -- 16x26x26 -> 32x24x24
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(32))

        addAndPrint(model, nn.SpatialConvolution(32, 32, 1, 1)) -- 32x24x24 -> 32x24x24
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(32))

        addAndPrint(model, nn.SpatialConvolution(32, 40, 3, 3)) -- 32x24x24 -> 40x22x22
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(40))

        addAndPrint(model, nn.SpatialConvolution(40, 64, 1, 1)) -- 40x22x22 -> 64x22x22
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(64))

        addAndPrint(model, nn.SpatialConvolution(64, 16, 3, 3)) -- 64x22x22 -> 16x20x20
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(16))
		
        addAndPrint(model, nn.SpatialConvolution(16, 16, 1, 1)) -- 16x20x20 -> 16x20x20
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(16))
		
        addAndPrint(model, nn.SpatialConvolution(16, 16, 3, 3)) -- 16x20x20 -> 16x18x18
	addAndPrint(model, nn.SpatialMaxPooling(2,2,2,2))       -- 16x18x18 -> 16x9x9
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(16))

	local lastSize = 9

        addAndPrint(model, nn.View(16*lastSize*lastSize):setNumInputDims(3))
        addAndPrint(model, nn.Linear(16*lastSize*lastSize, #classes))
        addAndPrint(model, nn.LogSoftMax())

elseif opt.network == 'n5' then

        print('Params', 'Layer')

        addAndPrint(model, nn.BatchFlip())
        addAndPrint(model, nn.SpatialConvolution(3, 32, 5, 5))  -- 3x32x32 -> 32x28x28
        addAndPrint(model, nn.SpatialMaxPooling(2,2,2,2))       -- 32x28x28 -> 32x14x14
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(32))

        addAndPrint(model, nn.SpatialConvolution(32, 64, 3, 3)) -- 32x14x14 -> 64x12x12
        addAndPrint(model, nn.SpatialMaxPooling(2,2,2,2))       -- 64x12x12 -> 64x6x6
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(64))

        addAndPrint(model, nn.SpatialConvolution(64, 64, 1, 1)) -- 64x6x6 -> 64x6x6
        addAndPrint(model, nn.ReLU(true))
        addAndPrint(model, nn.SpatialBatchNormalization(64))

        addAndPrint(model, nn.SpatialConvolution(64, 32, 3, 3))
        addAndPrint(model, nn.ReLU(true))

        addAndPrint(model, nn.View(32*4*4):setNumInputDims(3))
        addAndPrint(model, nn.Linear(32*4*4, #classes))
        addAndPrint(model, nn.LogSoftMax())

else
    print('Unknown model type')
    cmd:text()
    error()
end

model = model:cuda()
criterion = nn.ClassNLLCriterion():cuda()


w, dE_dw = model:getParameters()
print('== The whole model ==')
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'
image = require 'image'
local batchSize = opt.batch
local optimState = {}

--function dataAugmentation(x)

--end 

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)--:cuda()
		--x = dataAugmentation(x)
        local yt = labels:narrow(1, i, batchSize)--:cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
			if opt.optimization == "adam" then
				optim.adam(feval, w, optimState)
			elseif opt.optimization == "sgd" then
				optim.sgd(feval, w, optimState)
			elseif opt.optimization == "adagrad" then
				optim.adagrad(feval, w, optimState)
			else
				print('Unknown optimization')
				cmd:text()
				error()
			end
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end


epochs = opt.epoch
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
	print('Epoch ' .. e .. ':')
    print('Correctness (test): ' .. (1 - testError[e]) )
    if e % 5 == 0 then
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')

--  ****************************************************************
--  Network predictions
--  ****************************************************************

model:evaluate()   --turn off dropout

-- save the model
if opt.save == 1 then
    print('Save the model:' .. filename)
    torch.save(filename, model		)
end


--print(classes[testLabels[10]])
--[[print('')
print('testData[10]:size()')
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print('')
print('predicted:exp()')
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
]]

--  ****************************************************************
--  Visualizing Network Weights+Activations
--  ****************************************************************

--[[
local Weights_1st_Layer = model:get(1).weight
local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_1st_Layer,padding=2}),200)
saveTensorAsGrid(scaledWeights,'Weights_1st_Layer.jpg')

 
print('Input Image')
saveTensorAsGrid(testData[100],'testImg100.jpg')
model:forward(testData[100]:view(1,3,32,32):cuda())
for l=1,9 do
  print('Layer ' ,l, tostring(model:get(l)))
  local layer_output = model:get(l).output[1]
  saveTensorAsGrid(layer_output,'Layer'..l..'-'..tostring(model:get(l))..'.jpg')
  if ( l == 5 or l == 9 )then
	local Weights_lst_Layer = model:get(l).weight
	local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_lst_Layer[1],padding=2}),200)
	saveTensorAsGrid(scaledWeights,'Weights_'..l..'st_Layer.jpg')
  end 
end

]]

if opt.filename ~= "none" then
	local res = 1-testError[opt.epoch]
	print('Writing result ' .. res ..' to file: ' .. opt.filename)
	
	local file = io.open(opt.filename, "w")		-- Opens a file in append mode
	file:write(res)								-- appends a word test to the last line of the file
	file:close()								-- closes the open file
end



