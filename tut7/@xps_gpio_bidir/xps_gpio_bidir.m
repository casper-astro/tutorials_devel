%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://seti.ssl.berkeley.edu/casper/                                      %
%   Copyright (C) 2006 University of California, Berkeley                     %
%                                                                             %
%   This program is free software; you can redistribute it and/or modify      %
%   it under the terms of the GNU General Public License as published by      %
%   the Free Software Foundation; either version 2 of the License, or         %
%   (at your option) any later version.                                       %
%                                                                             %
%   This program is distributed in the hope that it will be useful,           %
%   but WITHOUT ANY WARRANTY; without even the implied warranty of            %
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             %
%   GNU General Public License for more details.                              %
%                                                                             %
%   You should have received a copy of the GNU General Public License along   %
%   with this program; if not, write to the Free Software Foundation, Inc.,   %
%   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.               %
%                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function b = xps_gpio(blk_obj)

%Is the block under consideration tagged as an xps block?
if ~isa(blk_obj,'xps_block')
    error('XPS_GPIO class requires a xps_block class object');
end

% Is the block under consideration a gpio_bidir block? 
if ~strcmp(get(blk_obj,'type'),'xps_gpio_bidir')
    error(['Wrong XPS block type: ',get(blk_obj,'type')]);
end

% Grab the block name. It will be handy when we need to grab parameters
blk_name = get(blk_obj,'simulink_name');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get the hardware platform (i.e. ROACH board, etc, and GPIO bank used by the block
xsg_obj = get(blk_obj,'xsg_obj');
hw_sys_full =  get(xsg_obj,'hw_sys');
hw_sys = strtok(hw_sys_full,':') %hw_sys_full is ROACH:SX95t (We only want "ROACH")
gpio_bank = get_param(blk_name,'bank')

%set the properties of the XSG object
s.hw_sys = hw_sys
s.io_group= gpio_bank
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%constructor
b = class(s,'xps_gpio_bidir',blk_obj);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start setting parameters needed to properly instantiate the pcore and modify the ucf file

% ip name
%The is the name of your pcore which will be instantiated when the toolflow sees the yellow block
b = set(b, 'ip_name','gpio_bidir');

% external ports
% Here we set up the external (i.e. to FPGA pin) connections required by the yellow block pcore

s.hw_sys
switch s.hw_sys
    case 'ROACH'
        switch s.io_group
          % The ROACH has GPIO banks operating at different voltages.
          case 'gpiob'
            iostandard = 'LVCMOS15';
          otherwise
            iostandard = 'LVCMOS25';
        end               
    % you might like to insert conditions here for other platforms
    otherwise
        iostandard = 'LVCMOS25';
end % switch 'hw_sys'

%io direction is inout
s.io_dir = 'inout'
bus_width = 8

ucf_fields = {};
ucf_values = {};

%ucf_fields = [ucf_fields, 'IOSTANDARD', termination];
%ucf_values = [ucf_values, iostandard, ''];

ucf_constraints = struct('IOSTANDARD',iostandard);

% Give a name to the external port name and buffer names. The iobname should match an entry in the hw_routes table,
% allowing the tools to map the name to an FPGA pin number.
% the extportname is the name given to the signal connected to the pins. It should be connected somewhere else in your pcore
% (see the last few lines of this script for the connection to the pcore)
% Be careful to make sure your signal names won't clash if you use multiple copies of the same yellow block in your design.
% here we use a modification the block name (which simulink mandates is unique)
extportname = [clear_name(blk_name), '_ext'];
iobname = [s.hw_sys, '.', s.io_group];

%this string is passed to the external port structure 
pin_str = ['{',iobname,'{[',num2str([1:bus_width]),']}}']



%ucf_constraints = cell2struct(ucf_values, ucf_fields, length(ucf_fields));

ext_ports.dio_buf =   {bus_width  s.io_dir  extportname  pin_str  'vector=true'  struct()  ucf_constraints};

b = set(b,'ext_ports',ext_ports);

% parameters
% These are the parameters that get passed to your pcore, which in turn set the parameters/generics defined in your HDL

%This block has no parameters
%b = set(b,'parameters',parameters);

% misc ports
% These are ports on your pcore that are not connected within your simulink design via the yellow block gateway ins/outs.
% Often a clock will be one of these signals
% In our GPIO birectional block, we also need to connect the pcore to the pins we specified above, using the extportname variable

%in our biderectional GPIO block, the block should be clocked by the same clock as the rest of the simulink design.

%find out the simulink clock source set by the user
xsg_obj = get(blk_obj,'xsg_obj');
simulink_clock =  get(xsg_obj,'clk_src');

%create the clock port called 'clk'
misc_ports.clk = {1 'in' simulink_clock};

b = set(b,'misc_ports',misc_ports);
