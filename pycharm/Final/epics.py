from softioc import builder

class Builder:
    '''
    --- builder ---
    softioc class responsible for creating IOC
    since there can only be one builder, the builder has to be created separately from the CamIOC not from each CamDatEps
    the distinction between the CamDatEps objects is made by an device name added to the PV name
    TO DO: if every CamDatEps object gets its own builder, the softioc module might have to be adjusted
    '''
    def __init__(self, builder_name):
        builder.SetDeviceName(builder_name)

    def start(self):
        # Boilerplate get the IOC started
        builder.LoadDatabase()


class Epics:
    '''
    --- epics ---
    epics interface for CamDatEps
    creates and updates PV records
    initial_control_param_values: set by DataAnalyzer
    control params -> parameter defined by user for data analysis
    fit params -> resulting parameter from fit analysis
    '''
    def __init__(self, cam_dat_eps, initial_control_param_values, init_dict=None):
        self.cam_dat_eps = cam_dat_eps

        # get init values from init_dict
        device_name, control_param_names, fit_param_names, error_names = self.load_names(init_dict)

        ### Datanalyzer records
        #roi boundaries
        self.ao_roi_x_start = builder.aOut(device_name +'_' + control_param_names['roi_x_start'],
                                           initial_value=initial_control_param_values['roi_x_start'],
                                           on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_x_start', v),
                                           always_update=True)
        self.ao_roi_x_stop = builder.aOut(device_name +'_' + control_param_names['roi_x_stop'],
                                          initial_value=initial_control_param_values['roi_x_stop'],
                                          on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_x_stop', v),
                                          always_update=True)
        self.ao_roi_y_start = builder.aOut(device_name +'_' + control_param_names['roi_y_start'],
                                           initial_value=initial_control_param_values['roi_y_start'],
                                           on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_y_start', v),
                                           always_update=True)
        self.ao_roi_y_stop = builder.aOut(device_name +'_' + control_param_names['roi_y_stop'],
                                          initial_value=initial_control_param_values['roi_y_stop'],
                                          on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_y_stop', v),
                                          always_update=True)

       #fit_area
        self.ao_factor = builder.aOut(device_name +'_' + control_param_names['factor'],
                                      initial_value=float(initial_control_param_values['factor']),
                                      on_update_name=lambda v, n: self.on_control_params_update('fit_area', 'factor', v),
                                      always_update=True)
        self.ao_threshold = builder.aOut(device_name +'_' + control_param_names['threshold'],
                                         initial_value=initial_control_param_values['threshold'],
                                         on_update_name=lambda v, n: self.on_control_params_update('fit_area', 'threshold', v),
                                         always_update=True)
        self.ao_median_flt = builder.boolOut(device_name +'_' + control_param_names['median_flt'],
                                         initial_value=initial_control_param_values['median_flt'],
                                         on_update_name=lambda v, n: self.on_control_params_update('fit_area', 'median_flt', v),
                                         always_update=True)

       #gauss_model
        self.ao_sampled = builder.aOut(device_name +'_' + control_param_names['sampled'],
                                       initial_value=initial_control_param_values['sampled'],
                                       on_update_name=lambda v, n: self.on_control_params_update('g_fit', 'sampled', v),
                                       always_update=True)

       #fit parameter
        self.ai_amplitude = builder.aIn(device_name+'_'+fit_param_names['amplitude'], initial_value=0)
        self.ai_center_x = builder.aIn(device_name+'_'+fit_param_names['center_x'], initial_value=0)
        self.ai_center_y = builder.aIn(device_name+'_'+fit_param_names['center_y'], initial_value=0)
        self.ai_sigma_x = builder.aIn(device_name+'_'+fit_param_names['sigma_x'], initial_value=0)
        self.ai_sigma_y = builder.aIn(device_name+'_'+fit_param_names['sigma_y'], initial_value=0)
        self.ai_rotation = builder.aIn(device_name+'_'+fit_param_names['rotation'], initial_value=0)
        self.ai_offset = builder.aIn(device_name+'_'+fit_param_names['offset'], initial_value=0)

        ###Errors
        self.ai_error_ia = builder.stringIn(device_name+'_'+error_names['error_ia'], initial_value='error1')
        self.ai_error_da = builder.stringIn(device_name+'_'+error_names['error_da'], initial_value='error2')

        ###Save settings
        self.ao_save_to_file_name = builder.stringOut(device_name+'_'+'SAVE',
                                       initial_value='PATH\FILE NAME',
                                       on_update=lambda v: self.on_save_settings(v),
                                       always_update=True)

        ###Camera settings change
        self.ao_cam_exposure_time = builder.stringOut(device_name + '_' + 'EXPOSURE_TIME_ABS',
                                       initial_value=str(self.cam_dat_eps.get_current_ia_feature('ExposureTimeAbs')),
                                       on_update=lambda v: self.on_ia_feature_update('ExposureTimeAbs', int(v)),
                                       always_update=True)

        #settings saved for later
        self.device_name, self.control_param_names, self.fit_param_names, self.error_names = device_name, control_param_names, fit_param_names, error_names


    def load_names(self, init_dict=None):
        default_device_name = "CAMERA"
        default_control_param_names = {'roi_x_start': 'AO_ROI_X_START',
                                       'roi_x_stop': 'AO_ROI_X_STOP',
                                       'roi_y_start': 'AO_ROI_Y_START',
                                       'roi_y_stop': 'AO_ROI_Y_STOP',
                                       'factor': 'AO_FACTOR',
                                       'threshold': 'AO_THRESHOLD',
                                       'median_flt': 'AO_MEDIAN_FLT',
                                       'sampled': 'AO_SAMPLED'}
        default_fit_param_names = {'amplitude': 'AI_AMPLITUDE',
                                   'center_x': 'AI_CENTER_X',
                                   'center_y': 'AI_CENTER_Y',
                                   'sigma_x': 'AI_SIGMA_X',
                                   'sigma_y': 'AI_SIGMA_Y',
                                   'rotation': 'AI_ROTATION',
                                   'offset': 'AI_OFFSET'}
        default_error_names = {'error_ia':"AI_ERROR_IA",
                               'error_da':"AI_ERROR_DA"}

        device_name = default_device_name
        control_param_names = default_control_param_names
        fit_param_names = default_fit_param_names
        error_names = default_error_names

        # if there is a new setting from init_dict change it
        if init_dict:
            if 'device_name' in init_dict:
                device_name = init_dict['device_name']
            if 'control_params' in init_dict:
                for param, name in init_dict['control_params'].items():
                    control_param_names[param] = name
            if 'fit_params' in init_dict:
                for param, name in init_dict['fit_params'].items():
                    fit_param_names[param] = name
            if 'error_names' in init_dict:
                error_names = init_dict['error_name']

        return device_name, control_param_names, fit_param_names, error_names


    def on_control_params_update(self, area, control_param_name, value):
        self.cam_dat_eps.on_control_params_update(area, control_param_name, value)
        print("in ", area, control_param_name, " change to ", value)

    def on_ia_feature_update(self, feature_name, value):
        self.cam_dat_eps.set_ia_feature(feature_name, value)

    def on_save_settings(self, file_name):
        self.cam_dat_eps.save_toJson(file_name)

    def get_epics_settings(self):
        # for saving
        settings = {'device_name': self.device_name, 'control_params': self.control_param_names,
                    'fit_params': self.fit_param_names, 'error_names': self.error_names}
        return settings

    def set_fit_params(self):
        param_list = self.cam_dat_eps.get_current_params()
        # order of params: [amplitude, centerx, centery, sigmax, sigmay, rot, offset]
        if param_list != []:
            self.ai_amplitude.set(param_list[0])
            self.ai_center_x.set(param_list[1])
            self.ai_center_y.set(param_list[2])
            self.ai_sigma_x.set(param_list[3])
            self.ai_sigma_y.set(param_list[4])
            self.ai_rotation.set(param_list[5])
            self.ai_offset.set(param_list[6])
        else:
            print("Empty param list")

    def set_error(self):
            # TO DO: find out WHY strings longer than 39 characters aren't supported
        error_message_ia, error_message_da = self.cam_dat_eps.get_errors()
        if error_message_ia:
            if len(error_message_ia) > 39:
                error_message_ia = error_message_ia[:39]
            self.ai_error_ia.set(error_message_ia)
        else:
            self.ai_error_da.set('no ia error')
        if error_message_da:
            if len(error_message_da) > 39:
                error_message_da = error_message_da[:39]
            self.ai_error_da.set(error_message_da)
        else:
            self.ai_error_da.set('no da error')

    def run(self):
        self.set_fit_params()
        print("new params set")
        self.set_error()
        # print("set error")


if __name__ == '__main__':

    init_dict_example = \
        {"device_name": "ALMUT",
         "control_params":
             {'roi_x_start': 'X_START',
              'roi_x_stop': 'X_STOP',
              'roi_y_start': 'Y_START',
              'roi_y_stop': 'Y_STOP',
              'factor': 'FACTOR',
              'threshold': 'THRESHOLD',
              'median_flt': 'MEDIAN_FLT',
              'sampled': 'SAMPLED'},
         "fit_params":
             {'amplitude': 'AMPLITUDE',
            'center_x': 'CENTER_X',
            'center_y': 'CENTER_Y',
            'sigma_x': 'SIGMA_X',
            'sigma_y': 'SIGMA_Y',
            'rotation': 'ROT',
            'offset': 'SET'}
         }


    initial_ctr_param_values = {'roi_x_start': 1,
                                       'roi_x_stop': 1,
                                       'roi_y_start': 1,
                                       'roi_y_stop': 1,
                                       'factor': 1,
                                       'threshold': 1,
                                       'median_flt': 1,
                                       'sampled': 1}

    #check if init okay
    epics = Epics(None, initial_ctr_param_values, init_dict_example)