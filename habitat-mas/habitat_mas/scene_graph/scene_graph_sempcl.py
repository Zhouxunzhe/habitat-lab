


class SceneGraphRtabmap(SceneGraphBase):
    # layers #
    object_layer = None
    region_layer = None
    
    # TODO: finetune the DBSCAN parameters
    def __init__(self, rtabmap_pcl, point_features=False, label_mapping=None, 
            scene_bounds=None, grid_map=None, map_resolution=0.05, dbscan_eps=1.0, 
            dbscan_min_samples=5, dbscan_num_processes=4, min_points_filter=5,
            dbscan_verbose=False, dbscan_vis=False, label_scale=2, 
            nms=True, nms_th=0.4):

        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = scene_bounds
        self.grid_map = grid_map
        self.map_resolution = map_resolution
        self.point_features = point_features
        self.object_layer = ObjectLayer()
        self.region_layer = RegionLayer()
        
        if self.grid_map is not None:
            self.region_layer.init_map(
                self.scene_bounds, self.map_resolution, self.grid_map
            )

        # 2. use DBSCAN with label as fourth dimension to cluster instance points
        
        points = ros_numpy.point_cloud2.pointcloud2_to_array(rtabmap_pcl)
        points = ros_numpy.point_cloud2.split_rgb_field(points)
        xyz = np.vstack((points["x"], points["y"], points["z"])).T
        # rgb = np.vstack((points["r"], points["g"], points["b"])).T
        # use g channel to store label 
        g = points["g"].T
        num_class = len(coco_categories)
        # cvrt from 0 for background to -1 for background
        class_label = np.round(g * float(num_class + 1) / 255.0).astype(int) - 1
        # filter out background points 
        objects_mask = (class_label >= 0)
        if not np.any(objects_mask): # no object points in semantic mesh 
            # stop initialization with empty scene graph
            return 
        objects_xyz = xyz[objects_mask]
        objects_label = class_label[objects_mask]
        sem_points = np.concatenate(
            (objects_xyz, label_scale * objects_label.reshape(-1, 1)), axis=1)

        # cluster semantic point clouds to object clusters 
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, 
                    n_jobs=dbscan_num_processes).fit(sem_points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        inst_labels = db.labels_
        object_ids = set(inst_labels)
        if dbscan_verbose:
            num_clusters = len(object_ids) - (1 if -1 in inst_labels else 0)
            num_noise = (inst_labels == -1).sum()
            print(f"DBSCAN on semantic point clouds, num_clusters ({num_clusters}), num_noise ({num_noise})")
        
        if dbscan_vis:
            pass
        
        # 3. non-maximum suppression: filter out noisy detection result 

        if nms:
            valid_object_ids = []
            valid_object_score_bboxes = [] # [p, x, y, z, l,w,h]
            for obj_id in object_ids:
                if obj_id == -1: # outliers
                    continue
                obj_xyz = objects_xyz[inst_labels == obj_id]
                if obj_xyz.shape[0] > min_points_filter:
                    label_modes, _ = stats.mode(objects_label[inst_labels == obj_id], nan_policy="omit")
                    # select mode label as object label
                    obj_label = label_modes[0]
                    if obj_label < len(label_mapping):
                        obj_cls_name = label_mapping[obj_label]
                        center = np.mean(obj_xyz, axis=0)
                        size = np.max(obj_xyz, axis=0) - np.min(obj_xyz, axis=0)
                        score_bbox = np.array([obj_xyz.shape[0], # num of points  
                                            center[0], center[1], center[2],
                                            size[0], size[1], size[2],
                                            ])
                        valid_object_ids.append(obj_id)
                        valid_object_score_bboxes.append(score_bbox)
            
            object_ids = valid_object_ids
            # there could be no valid objects founded 
            if len(valid_object_ids) > 0:
                valid_object_score_bboxes = np.stack(valid_object_score_bboxes, axis=0)
                selected_indices, _ = NMS(valid_object_score_bboxes, nms_th)
                object_ids = [valid_object_ids[idx] for idx in selected_indices]
                
        # 4. create object nodes in scene graph 
        
        for obj_id in object_ids:
            
            if obj_id == -1: # outliers
                continue
            
            obj_xyz = objects_xyz[inst_labels == obj_id]
            if obj_xyz.shape[0] > min_points_filter:
                label_modes, _ = stats.mode(objects_label[inst_labels == obj_id], nan_policy="omit")
                # select mode label as object label
                obj_label = label_modes[0]
                obj_cls_name = ""
                if obj_label >= 0:
                    obj_cls_name = label_mapping[obj_label]
                # else:
                #     obj_cls_name = "background"
                    # use axis-aligned bounding box for now 
                    center = np.mean(obj_xyz, axis=0)
                    rot_quat = np.array([0, 0, 0, 1])  # identity transform
                    size = np.max(obj_xyz, axis=0) - np.min(obj_xyz, axis=0)
                    if not self.point_features:
                        object_vertices = None
                    else:
                        object_vertices = obj_xyz
                        
                    object_node = self.object_layer.add_object(
                        center,
                        rot_quat,
                        size,
                        id=obj_id,
                        class_name=obj_cls_name,
                        label=obj_label,
                        vertices=object_vertices   
                    )

                # no region prediction module implemented 
                # connect object to region
                # region_node.add_object(object_node)

        return

    # def get_full_graph(self):
    #     """Return the full scene graph"""
    #     return None

    # def sample_graph(self, method, *args, **kwargs):
    #     """Return the sub-sampled scene graph"""
        
    #     return None

