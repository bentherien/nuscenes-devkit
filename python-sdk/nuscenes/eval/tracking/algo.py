"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import os
from typing import List, Dict, Callable, Tuple
import unittest

import numpy as np
import sklearn
import tqdm

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')

from nuscenes.eval.tracking.constants import MOT_METRIC_MAP, TRACKING_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricData
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.eval.tracking.render import TrackingRenderer
from nuscenes.eval.tracking.utils import print_threshold_metrics, create_motmetrics

# Using to get frame tokens
from nuscenes.nuscenes import NuScenes

class TrackingEvaluation(object):
    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int,
                 metric_worst: Dict[str, float],
                 verbose: bool = True,
                 output_dir: str = None,
                 render_classes: List[str] = None,
                 track_errors: bool = False,
                 nuscenes_info: NuScenes = None):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param metric_worst: Mapping from metric name to the fallback value assigned if a recall threshold
            is not achieved.
        :param verbose: Whether to print to stdout.
        :param output_dir: Output directory to save renders.
        :param render_classes: Classes to render to disk or None.

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
        """
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.class_name = class_name
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.num_thresholds = num_thresholds
        self.metric_worst = metric_worst
        self.verbose = verbose
        self.output_dir = output_dir
        self.render_classes = [] if render_classes is None else render_classes
        self.track_errors = track_errors
        self.error_events = {}
        self.errors_of_interest = ["SWITCH", "MISS", "FP"]
        self.error_type_to_count = {
            "SWITCH": "num_switch", 
            "MISS": "num_miss", 
            "FP": "num_fp",
        }
        self.frag_events = {}
        self.ids_events = {}
        self.pos_neg_pairs_events = {}
        self.positives_pairs = []
        self.negatives_pairs = []
        self.reidentification_error_pairs = {}

        self.n_scenes = len(self.tracks_gt)
        self.nusc = nuscenes_info

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def accumulate(self) -> TrackingMetricData:
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # Init.
        if self.verbose:
            print('Computing metrics for class %s...\n' % self.class_name)
        accumulators = []
        thresh_metrics = []
        md = TrackingMetricData()

        # Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # Register mot metrics.
        mh = create_motmetrics()

        # Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.

        thresholds, recalls = self.compute_thresholds(gt_box_count)
        print('\nthresholds:',thresholds, '\nrecalls:',recalls)

        md.confidence = thresholds
        md.recall_hypo = recalls
        if self.verbose:
            print('Computed thresholds\n')
        for t, threshold in enumerate(thresholds):
            # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue

            # Do not compute the same threshold twice.
            # This becomes relevant when a user submits many boxes with the exact same score.
            if threshold in thresholds[:t]:
                continue

            # Accumulate track data.
            acc, _ = self.accumulate_threshold(threshold)
            accumulators.append(acc)

            # Compute metrics for current threshold.
            thresh_name = self.name_gen(threshold)
            thresh_summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name)
            thresh_metrics.append(thresh_summary)

            # Print metrics to stdout.
            if self.verbose:
                print_threshold_metrics(thresh_summary.to_dict())

        # Concatenate all metrics. We only do this for more convenient access.
        if len(thresh_metrics) == 0:
            summary = []
        else:
            summary = pandas.concat(thresh_metrics)

        # Get the number of thresholds which were not achieved (i.e. nan).
        unachieved_thresholds = np.array([t for t in thresholds if np.isnan(t)])
        num_unachieved_thresholds = len(unachieved_thresholds)

        # Get the number of thresholds which were achieved (i.e. not nan).
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        num_duplicate_thresholds = len(valid_thresholds) - len(np.unique(valid_thresholds))

        # Sanity check.
        assert num_unachieved_thresholds + num_duplicate_thresholds + len(thresh_metrics) == self.num_thresholds

        # Figure out how many times each threshold should be repeated.
        rep_counts = [np.sum(thresholds == t) for t in np.unique(valid_thresholds)]

        # Store all traditional metrics.
        for (mot_name, metric_name) in MOT_METRIC_MAP.items():
            print('\nmot_name',mot_name)
            print('metric_name',metric_name)

            # Skip metrics which we don't output.
            if metric_name == '':
                continue

            # Retrieve and store values for current metric.
            if len(thresh_metrics) == 0:
                # Set all the worst possible value if no recall threshold is achieved.
                worst = self.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == 'ml':
                        worst = len(gt_track_ids)
                    elif metric_name in ['gt', 'fn']:
                        worst = gt_box_count
                    elif metric_name in ['fp', 'ids', 'frag']:
                        worst = np.nan  # We can't know how these error types are distributed.
                    else:
                        raise NotImplementedError

                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)

                # If a threshold occurred more than once, duplicate the metric values.
                assert len(rep_counts) == len(values)
                values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])

                # Pad values with nans for unachieved recall thresholds.
                all_values = [np.nan] * num_unachieved_thresholds
                all_values.extend(values)


            print('\nall_values',all_values)
            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        return md

    def accumulate_threshold(self, threshold: float = None) -> Tuple[pandas.DataFrame, List[float]]:
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        bens_frameid_to_timestamp = {}

        accs = []
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.
        thresh_name = None
        if threshold is not None:
            thresh_name = self.name_gen(threshold)
            self.frag_events[thresh_name] = []
            self.ids_events[thresh_name] = []
            self.pos_neg_pairs_events[thresh_name] = []
            self.positives_pairs = []
            self.negatives_pairs = []
            self.reidentification_error_pairs[thresh_name] = {}

        # Get mapping from scene_id to NuScenes Info index
        if self.track_errors:
            scene_id_to_nusc_idx = {}
            nusc_idx_to_scene_idx = []
            i = 0
            for scene in self.nusc.scene:
                scene_id_to_nusc_idx[scene['token']] = i
                nusc_idx_to_scene_idx.append(scene['token'])
                i += 1

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        for scene_id in tqdm.tqdm(self.tracks_gt.keys(), disable=not self.verbose, leave=False):

            # Initialize accumulator and frame_id for this scene
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes
            sample_token_list = []

            if self.track_errors:
                nusc_idx = scene_id_to_nusc_idx[scene_id]
                sample_token = self.nusc.scene[nusc_idx]['first_sample_token']

            # Setting up error monitoring
            if threshold is not None:
                if thresh_name not in self.error_events.keys():
                    self.error_events[thresh_name] = {}
                self.error_events[thresh_name][scene_id] = {}
                self.error_events[thresh_name][scene_id]["frame_errors"] = []
                prev_matches_H_to_O = {} # for positive, negative pairs in IDS
                prev_matches_O_to_H = {}
                for error_type in self.errors_of_interest:
                    total_metric = self.error_type_to_count[error_type]
                    self.error_events[thresh_name][scene_id][total_metric] = 0

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            if self.class_name in self.render_classes and threshold is None:
                save_path = os.path.join(self.output_dir, 'render', str(scene_id), self.class_name)
                os.makedirs(save_path, exist_ok=True)
                renderer = TrackingRenderer(save_path)
            else:
                renderer = None

            for timestamp in scene_tracks_gt.keys():
                bens_frameid_to_timestamp[len(bens_frameid_to_timestamp)] = timestamp

                # Select only the current class.
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]
                frame_gt = [f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name]

                # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                if threshold is not None:
                    frame_pred = [f for f in frame_pred if f.tracking_score >= threshold]

                # Abort if there are neither GT nor pred boxes.
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                    pred_boxes = np.array([b.translation[:2] for b in frame_pred])
                    distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

                # Distances that are larger than the threshold won't be associated.
                assert len(distances) == 0 or not np.all(np.isnan(distances))
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

                # Store scores of matches, which are used to determine recall thresholds.
                if threshold is None:
                    events = acc.events.loc[frame_id]
                    matches = events[events.Type == 'MATCH']
                    match_ids = matches.HId.values
                    match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                    scores.extend(match_scores)
                else:
                    events = None

                # Record errors of interest for analysis
                if self.track_errors and threshold is not None:
                    frame_errors = {}
                    for error_type in self.errors_of_interest:
                        events = acc.events.loc[frame_id]
                        errors = events[events.Type == error_type]
                        frame_errors[error_type] = []
                        total_metric = self.error_type_to_count[error_type]
                        if not errors.empty:
                            """
                            - `Type` one of `('MATCH', 'SWITCH', 'MISS', 'FP', 'RAW')`
                            - `OId` object id or np.nan when `'FP'` or `'RAW'` and object is not present
                            - `HId` hypothesis id or np.nan when `'MISS'` or `'RAW'` and hypothesis is not present
                            - 'D` distance or np.nan when `'FP'` or `'MISS'` or `'RAW'` and either object/hypothesis is absent
                            """
                            frame_errors['frame_id'] = frame_id
                            frame_errors['sample_token'] = sample_token
                            for _, error in errors.iterrows():
                                
                                error_entry = {'type': error.Type,
                                                'object_id': error.OId,
                                                'hypothesis_id': error.HId, 
                                                'distance': error.D, 
                                                }
                                
                                # If the type has a ground truth match, include details of GT
                                if error_type in ['SWITCH', 'MISS']:
                                    # Get Ground Truth
                                    for gt in frame_gt:
                                        if gt.tracking_id == error.OId:
                                            error_entry["gt"] = {
                                                "num_pts" : int(gt.num_pts),
                                                "size" : list(gt.size),
                                                "rotation" : list(gt.rotation),
                                                "velocity" : list(gt.velocity),
                                                "translation" : list(gt.translation),
                                                "ego_translation" : list(gt.ego_translation),
                                                "ego_dist" : float(gt.ego_dist),
                                                "tracking_name" : gt.tracking_name,
                                                "tracking_score" : float(gt.tracking_score)                                          
                                                }

                                frame_errors[error_type].append(error_entry)
                                self.error_events[thresh_name][scene_id][total_metric] += 1
                            
                    self.error_events[thresh_name][scene_id]["frame_errors"].append(frame_errors)
                
                # Look out for Positive-Negative Pairs for IDS
                if self.track_errors and threshold is not None:
                    events = acc.events.loc[frame_id]
                    matches = events[events.Type == "MATCH"]
                    if frame_id > 0:
                        switches = events[events.Type == "SWITCH"]
                        for _, switch in switches.iterrows():
                            curr_obj = switch.OId
                            prev_gt = acc.events.loc[frame_id-1]['OId'].unique()
                            curr_gt = acc.events.loc[frame_id]['OId'].unique()
                            # Check if the hypothesis was matched in previous frame, both objects exist in consecutive frames
                            if int(switch.HId) in prev_matches_H_to_O.keys() and switch.OId in prev_gt \
                                and str(prev_matches_H_to_O[int(switch.HId)]) in curr_gt:
                                
                                # Special Case: 2 Ground Truths Swap Hypotheses
                                hypothesis_switch = False
                                # Check if the current object was in the previous frame
                                if str(switch.OId) in prev_matches_O_to_H.keys():
                                    # "Other Ground Truth" Object ID, Hypothesis in previous frame
                                    prev_hypo2 = int(prev_matches_O_to_H[str(switch.OId)])
                                    # Check if the other ground truth also incurred a switch
                                    if not switches[switches.OId == prev_matches_H_to_O[int(switch.HId)]].empty:
                                        curr_hypo2 = int(switches[switches.OId == prev_matches_H_to_O[int(switch.HId)]].HId.values[0])
                                        # Check if the hypothesis is same on this frame and last frame for the"Other Ground Truth"
                                        if prev_hypo2 == curr_hypo2:
                                            hypothesis_switch = True

                                neg_pos_pair = {
                                    'scene_id': str(scene_id),
                                    'negative' : {
                                        'hypothesis_id': int(switch.HId),
                                        'curr_object_id': str(switch.OId),
                                        'curr_sample_token': str(sample_token),
                                        'prev_object_id': str(prev_matches_H_to_O[int(switch.HId)]),
                                        'prev_sample_token': str(sample_token_list[-1]),
                                    },
                                    'curr_positive':{
                                        'object_id': str(switch.OId),
                                        'curr_sample_token': str(sample_token),
                                        'prev_sample_token': str(sample_token_list[-1]),
                                    },
                                    'prev_positive':{
                                        'object_id': str(prev_matches_H_to_O[int(switch.HId)]),
                                        'curr_sample_token': str(sample_token),
                                        'prev_sample_token': str(sample_token_list[-1]),
                                    },
                                    'frame_num': int(frame_id),
                                    'prev_frame_num': int(frame_id-1),
                                    'class': self.class_name,
                                    'hypothesis_switch': hypothesis_switch
                                }

                                if hypothesis_switch:
                                    neg_pos_pair['negative2'] = {
                                        'hypothesis_id': int(prev_hypo2),
                                        'curr_object_id': str(prev_matches_H_to_O[int(switch.HId)]),
                                        'curr_sample_token': str(sample_token),
                                        'prev_object_id': str(switch.OId),
                                        'prev_sample_token': str(sample_token_list[-1]),
                                    }

                                self.pos_neg_pairs_events[thresh_name].append(neg_pos_pair)
                                
                    # Update pairs for next frame
                    prev_matches_H_to_O.clear()
                    prev_matches_O_to_H.clear()
                    for _, match in matches.iterrows():
                        prev_matches_H_to_O[int(match.HId)] = match.OId
                        prev_matches_O_to_H[match.OId] = int(match.HId)

                # Render the boxes in this frame.
                if self.class_name in self.render_classes and threshold is None:
                    renderer.render(events, timestamp, frame_gt, frame_pred)

                # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1
                if self.track_errors:
                    sample_token_list.append(sample_token)
                    sample_token = self.nusc.get('sample', sample_token)['next']

            # Store Number of Fragmentations
            if self.track_errors and threshold is not None:
                events = acc.events
                gt_object_ids = acc.events['OId'].unique()
                for obj in gt_object_ids:
                    if obj == 'nan':
                        continue
                    obj_hist = acc.events[acc.events.Type != 'RAW'].loc[acc.events.OId == obj]
                    num_frag, frag_loc = self.check_for_fragmentation(obj_hist)
                    for frag_id in range(num_frag):
                        # store Scene Token, Object (Annotation) ID, Frame Occured
                        frame_num = int(frag_loc[frag_id])
                        frag_instance = {
                            'scene_id': scene_id,
                            'object_id': obj,
                            'sample_token': sample_token_list[frame_num],
                            'prev_sample_token': sample_token_list[frame_num-1],
                            'prev_hypothesis_id': obj_hist.HId[frame_num-1].values[0],
                            'frame_num': frame_num,
                            'class': self.class_name
                        }
                        self.frag_events[thresh_name].append(frag_instance)

            # Store Number of IDS
            if self.track_errors and threshold is not None:
                events = acc.events
                gt_object_ids = acc.events['OId'].unique()
                for obj in gt_object_ids:
                    if obj == 'nan':
                        continue
                    obj_hist = acc.events[acc.events.Type != 'RAW'].loc[acc.events.OId == obj]
                    num_ids, ids_loc, prev_match_loc = self.check_for_ids(obj_hist)
                    for ids_id in range(num_ids):
                        # store Scene Token, Object (Annotation) ID, Frame Occured
                        frame_num = int(ids_loc[ids_id])
                        prev_match_frame_num = int(prev_match_loc[ids_id])
                        ids_instance = {
                            'scene_id':str(scene_id),
                            'object_id': str(obj),
                            'sample_token': str(sample_token_list[frame_num]),
                            'hypothesis_id':int(obj_hist.HId[frame_num].values[0]),
                            'prev_match_sample_token': str(sample_token_list[prev_match_frame_num]),
                            'prev_match_hypothesis_id': int(obj_hist.HId[prev_match_frame_num].values[0]),
                            'frame_num': int(frame_num),
                            'prev_match_frame_num': int(prev_match_frame_num),
                            'from_frag': int((obj_hist.Type[frame_num-1] == 'MISS').values[0]), # if the prev frame was MISS, this SWITCH came from a fragmenetation
                            'class': str(self.class_name)
                        }
                        self.ids_events[thresh_name].append(ids_instance)
            
            if self.track_errors and threshold is not None and self.class_name == "car":
                negs, positives = self.get_neg_pos_pairs(acc.events, sample_token_list, scene_id)
                self.positives_pairs = self.positives_pairs + positives
                self.negatives_pairs = self.negatives_pairs + negs
            accs.append(acc)
        
        if self.track_errors and threshold is not None and self.class_name == "car":
            self.reidentification_error_pairs[thresh_name] = {
                'positives': self.positives_pairs,
                'negatives': self.negatives_pairs,
                'positive_count': len(self.positives_pairs),
                'negative_count': len(self.negatives_pairs)
            }
        # Find Positive and Negative Error Pairs
        

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)
        # acc_merged, ben_mappings = MOTAccumulatorCustom.merge_event_dataframes(accs, return_mappings=True)


        return acc_merged, scores

    def compute_thresholds(self, gt_box_count: int) -> Tuple[List[float], List[float]]:
        """
        Compute the score thresholds for predefined recall values.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_box_count: The number of GT boxes for this class.
        :return: The lists of thresholds and their recall values.
        """
        # Run accumulate to get the scores of TPs.
        _, scores = self.accumulate_threshold(threshold=None)

        # Abort if no predictions exist.
        if len(scores) == 0:
            return [np.nan] * self.num_thresholds, [np.nan] * self.num_thresholds

        # Sort scores.
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]

        # Compute recall levels.
        tps = np.array(range(1, len(scores) + 1))
        rec = tps / gt_box_count
        assert len(scores) / gt_box_count <= 1

        # Determine thresholds.
        max_recall_achieved = np.max(rec)
        rec_interp = np.linspace(self.min_recall, 1, self.num_thresholds).round(12)
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # Cast to list.
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # Reverse order for more convenient presentation.
        thresholds.reverse()
        rec_interp.reverse()

        # Check that we return the correct number of thresholds.
        assert len(thresholds) == len(rec_interp) == self.num_thresholds

        return thresholds, rec_interp

    def check_for_fragmentation(self, dfo):
        """
        Total number of switches from tracked to not tracked.
        """
        num_frag = 0
        loc_of_frag = [] # store the frame_id before and after frag occurs
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            return num_frag, loc_of_frag
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        num_frag = diffs[diffs == 1].count()
        rel_loc_of_frag = (np.where(diffs.to_numpy() == 1.0)[0]).tolist() # +1 since we cound the first MISS as the fragment location.
        multiidx = diffs.keys()
        loc_of_frag = [multiidx[idx][0] for idx in rel_loc_of_frag]
        return num_frag, loc_of_frag
    
    def check_for_ids(self, dfo):
        """
        Total number of track switches.
        """
        n_ids = dfo.Type.isin(["SWITCH"]).sum()
        if n_ids == 0:
            return n_ids, [], []
        
        # Get location of IDS
        rel_loc_of_ids = (np.where(dfo.Type == "SWITCH")[0]).tolist()
        _ = dfo.Type.apply(lambda x: 0)
        multiidx = _.keys()
        loc_of_ids = [multiidx[idx][0] for idx in rel_loc_of_ids]

        # Get location of last match for each IDS
        prev_matches_loc = []
        for ids_idx in loc_of_ids:
            prev = ids_idx - 1
            while dfo.loc[prev].Type.values[0] not in ['SWITCH', 'MATCH'] and prev >= multiidx[0][0]:
                prev = prev - 1
            prev_matches_loc.append(prev)
    
        return n_ids, loc_of_ids, prev_matches_loc

    def get_neg_pos_pairs(self, events, sample_token_list, scene_token):
        frame_ids = np.unique([x[0] for x in events.index])

        # Get all the unique hypothesis in the scene
        all_hypo = [events.loc[i]['HId'] for i in np.unique([x[0] for x in events.index])]
        all_hypo = [y for x in all_hypo for y in x if str(y) != 'nan']
        all_hypo = np.unique(all_hypo)

        # Get all the unique ground truth in the scene
        all_gt = [events.loc[i]['OId'] for i in np.unique([x[0] for x in events.index])]
        all_gt = [y for x in all_gt for y in x if str(y) != 'nan']
        all_gt = np.unique(all_gt)

        # Get all the negatives
        negatives = []
        for hypo in all_hypo:
            hypo_hist = events[events.Type != 'RAW'].loc[events.HId == hypo]
            OId_hist = hypo_hist['OId'].to_list()
            if len(OId_hist) == 1:
                continue
            prev_OId = OId_hist[0]

            for i in range(1, len(OId_hist)):
                curr_OId = OId_hist[i]
                # False Positive case
                if str(curr_OId) == 'nan' or str(prev_OId) == 'nan':
                    prev_OId = OId_hist[i]
                    continue
                # Check if there was a switch, if so save neg. pair instance
                if curr_OId != prev_OId:
                    neg_pair = {
                        'hypothesis_id': hypo,
                        'scene_token': scene_token,
                        'prev_sample_token': sample_token_list[i-1],
                        'prev_object_token': prev_OId,
                        'curr_sample_token': sample_token_list[i],
                        'curr_object_token': curr_OId,                        
                    }
                    negatives.append(neg_pair)
                prev_OId = OId_hist[i]

        # Get all the positives
        positives = []
        for obj in all_gt:
            obj_hist = events[events.Type != 'RAW'].loc[events.OId == obj]
            HId_hist = obj_hist['HId'].to_list()
            if len(HId_hist) == 1:
                continue
            prev_HId = HId_hist[0]
            last_non_nan_HId = [HId_hist[0], sample_token_list[0]] # store hypo, sample_token
            nan_counter = 0
            for i in range(1, len(HId_hist)):
                curr_HId = HId_hist[i]
                # Make sure we have matched an instance of this object before
                if str(last_non_nan_HId[0]) != 'nan': 
                    # False Negative case / Fragmentation
                    if str(curr_HId) == 'nan':
                        nan_id = 'nan'
                        if nan_counter > 0:
                            nan_id = 'nan{}'.format(nan_counter)

                        pos_pair = {
                            'object_token': obj,
                            'scene_token': scene_token,
                            'prev_sample_token': last_non_nan_HId[1],
                            'prev_hypothesis': last_non_nan_HId[0],
                            'curr_sample_token': sample_token_list[i],
                            'curr_hypothesis': nan_id,                        
                        }
                        positives.append(pos_pair)
                        nan_counter += 1
                        prev_OId = HId_hist[i]
                        continue
                    # Check if there was a switch, if so save pos. pair instance
                    if curr_HId != prev_HId:
                        pos_pair = {
                            'object_token': obj,
                            'scene_token': scene_token,
                            'prev_sample_token': last_non_nan_HId[1],
                            'prev_hypothesis': last_non_nan_HId[0],
                            'curr_sample_token': sample_token_list[i],
                            'curr_hypothesis': curr_HId,                        
                        }
                        positives.append(pos_pair)
                    
                prev_HId = HId_hist[i]
                if str(curr_HId) != 'nan':
                    last_non_nan_HId = [HId_hist[i], sample_token_list[i]]
                nan_counter = 0
        
        return negatives, positives