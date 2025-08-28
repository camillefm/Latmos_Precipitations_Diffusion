from src.Metrics.metrics_function import distance_metrics, anormality_metrics, replace_image
from src.Metrics.metrics_function import range_masks, update_dictionary, return_binary_rain, pixel_metrics, normalize_dictionary, normalize_dict_by_row
from src.Sampling.sample import sample

import torch

class EvalMetrics:
    def __init__(self, model, timesteps, normalized=False, log1p=False, number_of_images_plot =12):
        self.model = model
        self.timesteps = timesteps
        self.normalized = normalized
        self.log1p = log1p
        self.number_of_images_plot = number_of_images_plot

        # Determine model device
        self.device = next(model.parameters()).device

        self.reset()

    def update_batch(self, tb, r, rq):

        self.sampled_images = sample(self.model, tb, self.timesteps, self.normalized, self.log1p, init_noise=None, return_all_steps=False).to(self.device)

        #put negative values to 0
        self.sampled_images = torch.clamp(self.sampled_images, min=0)

        batch_size = tb.shape[0]

        for i in range(batch_size):
            rain_i = r[i]
            sample_i = self.sampled_images[i]

            # Move tensors to CPU and detach before metric computation
            rain_i = rain_i.detach().cpu()
            sample_i = sample_i.detach().cpu()

            #add pixel values to the lists for histogram plotting
            self.add_pixel_values(sampled_image=sample_i, rain_image=rain_i)

            distance_metrics_result = distance_metrics(rain=rain_i, sampled_image=sample_i)
            anormality_metrics_result = anormality_metrics(sampled_image=sample_i)
            pixel_metrics_result = pixel_metrics(self.rain_pixel_values, self.sampled_pixel_values)
            confusion = return_binary_rain(rain_i, sample_i, min_threshold_rain=self.model.valid_data_threshold,min_threshold_sampled=self.model.valid_data_threshold)
            # Compute metrics from masks
            self.compute_metrics_from_masks(rain_i, sample_i, rain_value_threshold=self.model.valid_data_threshold)

            
            update_dictionary(self.binary_metrics,  confusion)
            update_dictionary(self.distance_epoch, distance_metrics_result)
            update_dictionary(self.anormality_epoch, anormality_metrics_result)
            update_dictionary(self.pixel_metrics, pixel_metrics_result)

            for key, value in distance_metrics_result.items():
                if key != "early stopping rmse*(1-csi)":
                    list_best_metric = getattr(self, f'best_index_{key}')
                    list_best_images = getattr(self, f'best_images_{key}')
                    list_worst_metric = getattr(self, f'worst_index_{key}')
                    list_worst_images = getattr(self, f'worst_images_{key}')
                    tuple_image = (tb[i], rain_i, sample_i, rq[i])
                    replace_image(tuple_image=tuple_image, criterion=value, list_image=list_best_images,list_criterion=list_best_metric, criterion_name=key, order="best")
                    replace_image(tuple_image = tuple_image, criterion=value, list_image=list_worst_images,list_criterion=list_worst_metric, criterion_name=key, order="worst")
                
            replace_image(tuple_image=(tb[i], rain_i, sample_i, rq[i]), criterion=None, list_image=self.list_random_images, list_criterion=None, criterion_name=None, order="random")
            self.number_of_samples += 1

        return self.sampled_images
    

    def add_pixel_values(self, sampled_image, rain_image):
        """
        Add pixel values to the lists for histogram plotting.
        """
        if isinstance(sampled_image, torch.Tensor):
            sampled_image = sampled_image.detach().cpu().numpy()
        if isinstance(rain_image, torch.Tensor):
            rain_image = rain_image.detach().cpu().numpy()

        self.rain_pixel_values.extend(rain_image.flatten())
        self.sampled_pixel_values.extend(sampled_image.flatten())

        

    def compute_metrics_from_masks(self, rain, sampled_image, rain_value_threshold=0.1):
        rain_masks, sampled_masks, type_rain = range_masks(rain, sampled_image, rain_value_threshold)

        for rain_key in rain_masks.keys():
            if rain_key not in self.range_metrics:
                self.range_metrics[rain_key] = {}
                
            if rain_key not in self.nb_occurence_range:
                self.nb_occurence_range[rain_key] = 0

            if rain_key not in self.range_confusions:
                self.range_confusions[rain_key] = {}

            for sampled_key in sampled_masks.keys():
                r_masked = rain_masks[rain_key] # mask rain image
                s_masked = sampled_masks[sampled_key] # mask sampled image
                valid = ~torch.isnan(r_masked)

                if valid.sum().item() == 0:
                    tp = 0
                    confusion = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                else:
                    # Compute confusion matrix
                    confusion = return_binary_rain(
                        r_masked,
                        s_masked,
                        min_threshold_rain=type_rain[rain_key][0],
                        max_threshold_rain=type_rain[rain_key][1],
                        min_threshold_sampled=type_rain[sampled_key][0],
                        max_threshold_sampled=type_rain[sampled_key][1]
                    )
                    tp = confusion['tp']

                # Update per-range confusion counts
                if sampled_key not in self.range_confusions[rain_key]:
                    self.range_confusions[rain_key][sampled_key] = 0
                self.range_confusions[rain_key][sampled_key] += tp

                # Only compute metrics when comparing same rain/sampled category
                if rain_key == sampled_key and valid.sum().item() > 0:
                    self.nb_occurence_range[rain_key] += 1

                    distance_metrics_result = distance_metrics(
                        rain[valid],
                        sampled_image[valid],
                        rain_value_threshold,
                        metrics_type=['rmse', 'mae', 'csi', 'f1_score', 'accuracy', 'precision', 'recall'],
                        confusion=confusion  # You must handle this in distance_metrics()
                    )

                    update_dictionary(self.range_metrics[rain_key], distance_metrics_result)


                   
        
        

    def compute_metrics(self):
        normalize_dictionary(self.distance_epoch, self.number_of_samples)
        normalize_dictionary(self.binary_metrics, self.number_of_samples)
        normalize_dictionary(self.pixel_metrics, self.number_of_samples)

        self.range_confusions = normalize_dict_by_row(self.range_confusions)

        for range_key in self.range_metrics.keys():
            normalize_dictionary(self.range_metrics[range_key], self.nb_occurence_range[range_key])
        
            
        
        #remove all torch.tensor(0) from the lists
        self.list_random_images = [img for img in self.list_random_images if not (isinstance(img, torch.Tensor) and torch.equal(img, torch.tensor(0)))]

        return self.distance_epoch, self.anormality_epoch, self.binary_metrics, self.pixel_metrics


    def reset(self):
        self.distance_epoch = {}
        self.anormality_epoch = {}
        self.number_of_samples = 0
        
        self.range_metrics = {} #dictionary of dictionaries to store metrics for each range of rain

        self.range_confusions = {} #dictionary of dictionaries to store confusion matrices for each range of rain
        self.nb_occurence_range = {} #dictionary to store the number of occurences for each range of rain
        

        self.binary_metrics = { }  

        # Initialize lists for pixel values for histogram plotting
        self.rain_pixel_values = []
        self.sampled_pixel_values = []

        self.pixel_metrics = {}

        # Initialize lists for best images indexes
        self.best_index_rmse = [torch.tensor(999) for _ in range(self.number_of_images_plot)]
        self.best_index_mae = [torch.tensor(999) for _ in range(self.number_of_images_plot)]
        self.best_index_f1_score = [torch.tensor(0) for _ in range(self.number_of_images_plot)]
        self.best_index_accuracy = [torch.tensor(0) for _ in range(self.number_of_images_plot)]
        self.best_index_csi = [torch.tensor(0) for _ in range(self.number_of_images_plot)]
        self.best_index_precision = [torch.tensor(0) for _ in range(self.number_of_images_plot)]
        self.best_index_recall = [torch.tensor(0) for _ in range(self.number_of_images_plot)]

        # initialize lists for best images 
        self.best_images_rmse = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.best_images_mae = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.best_images_f1_score = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.best_images_accuracy = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.best_images_csi = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.best_images_precision = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.best_images_recall = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
          
        # initialize lists for worst images indexes
        self.worst_index_rmse = [torch.tensor(0) for _ in range(self.number_of_images_plot)]
        self.worst_index_mae = [torch.tensor(0) for _ in range(self.number_of_images_plot)]
        self.worst_index_f1_score = [torch.tensor(999) for _ in range(self.number_of_images_plot)]
        self.worst_index_accuracy = [torch.tensor(999) for _ in range(self.number_of_images_plot)]
        self.worst_index_csi = [torch.tensor(999) for _ in range(self.number_of_images_plot)]
        self.worst_index_precision = [torch.tensor(999) for _ in range(self.number_of_images_plot)]
        self.worst_index_recall = [torch.tensor(999) for _ in range(self.number_of_images_plot)]

        # initialize lists for worst images
        self.worst_images_rmse = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.worst_images_mae = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.worst_images_f1_score = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)] 
        self.worst_images_accuracy = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.worst_images_csi = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.worst_images_precision = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]
        self.worst_images_recall = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot)]

        # Initialize list for random images
        self.list_random_images = [tuple(torch.tensor(0) for _ in range(4)) for _ in range(self.number_of_images_plot*4)]
