function gpio_bidir_init(cursys)

% We need to rename the gateway blocks in the yellow block
% so that they respect the heirarchical naming conventions
% required by the toolflow

% find all the gateway in/out blocks
gateway_outs = find_system(cursys, ...
        'searchdepth', 1, ...
        'FollowLinks', 'on', ...
        'lookundermasks', 'all', ...
        'masktype','Xilinx Gateway Out Block'); 

gateway_ins = find_system(cursys, ...
        'searchdepth', 1, ...
        'FollowLinks', 'on', ...
        'lookundermasks', 'all', ...
        'masktype','Xilinx Gateway In Block');

%rename the gateway outs
for i =1:length(gateway_outs)
    gw = gateway_outs{i};
    gw_name = get_param(gw, 'Name');
    if regexp(gw_name, 'in_not_out_i$')
        set_param(gw, 'Name', clear_name([cursys, '_in_not_out_i']));
    elseif regexp(gw_name, 'din_i$')
        set_param(gw, 'Name', clear_name([cursys, '_din_i']));
    else 
        parent_name = get_param(gw, 'Parent');
        errordlg(['Unknown gateway: ', parent_name, '/', gw_name]);
    end
end 

%rename the gateway ins
for i =1:length(gateway_ins)
    gw = gateway_ins{i};
    gw_name = get_param(gw, 'Name');
    if regexp(gw_name, 'dout_o$')
        set_param(gw, 'Name', clear_name([cursys, '_dout_o']));
    else 
        parent_name = get_param(gw, 'Parent');
        errordlg(['Unknown gateway: ', parent_name, '/', gw_name]);
    end
end
