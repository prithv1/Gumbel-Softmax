#! /usr/bin/env lua
--
-- gum_softmax.lua
-- Copyright (C) 2016 prithv1 <prithv1@vt.edu>
--
-- Distributed under terms of the MIT license.
-- Add an nn.Log() layer before using this module


require 'nn'	
require 'torch'

torch.manualSeed(123)

local GumSoftmax, parent = torch.class('nn.GumSoftmax', 'nn.Module')

function GumSoftmax:__init(temperature, eps)
	parent.__init(self)
	self.temp = temperature
	self.eps = 1e-20 or eps
end

-- Sample gumbel distribution values from (0,1)
function sample_gumbel(shape, eps)
	local uniform_sample = torch.rand(tonumber(shape[1]), tonumber(shape[2]))
	local gumbel_sample = -torch.log(-torch.log(uniform_sample + eps) + eps)
	return gumbel_sample
end

-- Forward pass
function GumSoftmax:updateOutput(input)
	local input_size = input:size()
	self.gum_sample = sample_gumbel(input_size, self.eps)
	self.logits = (self.gum_sample + input) / self.temp
	self.softmax_module = nn.Sequential()
	self.softmax_module:add(nn.Identity()):add(nn.SoftMax())
	self.output = self.softmax_module:forward(self.logits)
	return self.output
end

-- Backward pass
function GumSoftmax:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input)
	self.gradInput:zero()
	local logits = (sample_gumbel(input:size(), self.eps) + input) / self.temp
	local softmax_module = nn.Sequential()
	softmax_module:add(nn.Identity()):add(nn.SoftMax())
	self.gradInput = softmax_module:backward(logits, gradOutput)
	self.gradInput = self.gradInput / self.temp
	return self.gradInput
end
