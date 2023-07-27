# Auto generated from model_card-schema.yaml by pythongen.py version: 0.9.0
# Generation date: 2023-07-26T14:06:54
# Schema: personinfo
#
# id: https://w3id.org/linkml/examples/personinfo
# description:
# license: https://creativecommons.org/publicdomain/zero/1.0/

import dataclasses
import re
from jsonasobj2 import JsonObj, as_dict
from typing import Optional, List, Union, Dict, ClassVar, Any
from dataclasses import dataclass
from linkml_runtime.linkml_model.meta import EnumDefinition, PermissibleValue, PvFormulaOptions

from linkml_runtime.utils.slot import Slot
from linkml_runtime.utils.metamodelcore import empty_list, empty_dict, bnode
from linkml_runtime.utils.yamlutils import YAMLRoot, extended_str, extended_float, extended_int
from linkml_runtime.utils.dataclass_extensions_376 import dataclasses_init_fn_with_kwargs
from linkml_runtime.utils.formatutils import camelcase, underscore, sfx
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from rdflib import Namespace, URIRef
from linkml_runtime.utils.curienamespace import CurieNamespace
from linkml_runtime.linkml_model.types import Boolean, Float, Integer, String
from linkml_runtime.utils.metamodelcore import Bool

metamodel_version = "1.7.0"
version = None

# Overwrite dataclasses _init_fn to add **kwargs in __init__
dataclasses._init_fn = dataclasses_init_fn_with_kwargs

# Namespaces
LINKML = CurieNamespace('linkml', 'https://w3id.org/linkml/')
PERSONINFO = CurieNamespace('personinfo', 'https://w3id.org/linkml/examples/personinfo')
DEFAULT_ = PERSONINFO


# Types

# Class references



@dataclass
class ModelCard(YAMLRoot):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = PERSONINFO.ModelCard
    class_class_curie: ClassVar[str] = "personinfo:ModelCard"
    class_name: ClassVar[str] = "ModelCard"
    class_model_uri: ClassVar[URIRef] = PERSONINFO.ModelCard

    organization: Optional[str] = None
    modelDate: Optional[str] = None
    modelVersion: Optional[float] = None
    modelType: Optional[str] = None
    trainingApproaches: Optional[str] = None
    moreInformation: Optional[str] = None
    citationDetails: Optional[str] = None
    contactInfo: Optional[str] = None
    primaryIntendedUses: Optional[str] = None
    primaryIntendedUsers: Optional[str] = None
    outOfScopeUseCases: Optional[str] = None
    relevantFactors: Optional[str] = None
    evaluationFactors: Optional[str] = None
    modelPerformanceMeasures: Optional[str] = None
    decisionThresholds: Optional[str] = None
    variationApproaches: Optional[str] = None
    evaluationDatasets: Optional[str] = None
    evaluationMotivation: Optional[str] = None
    evaluationPreprocessing: Optional[str] = None
    trainingDatasets: Optional[str] = None
    trainingMotivation: Optional[str] = None
    trainingPreprocessing: Optional[str] = None
    unitaryResults: Optional[str] = None
    intersectionalResults: Optional[str] = None
    ethicalConsiderations: Optional[str] = None
    caveatsAndRecommendations: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self.organization is not None and not isinstance(self.organization, str):
            self.organization = str(self.organization)

        if self.modelDate is not None and not isinstance(self.modelDate, str):
            self.modelDate = str(self.modelDate)

        if self.modelVersion is not None and not isinstance(self.modelVersion, float):
            self.modelVersion = float(self.modelVersion)

        if self.modelType is not None and not isinstance(self.modelType, str):
            self.modelType = str(self.modelType)

        if self.trainingApproaches is not None and not isinstance(self.trainingApproaches, str):
            self.trainingApproaches = str(self.trainingApproaches)

        if self.moreInformation is not None and not isinstance(self.moreInformation, str):
            self.moreInformation = str(self.moreInformation)

        if self.citationDetails is not None and not isinstance(self.citationDetails, str):
            self.citationDetails = str(self.citationDetails)

        if self.contactInfo is not None and not isinstance(self.contactInfo, str):
            self.contactInfo = str(self.contactInfo)

        if self.primaryIntendedUses is not None and not isinstance(self.primaryIntendedUses, str):
            self.primaryIntendedUses = str(self.primaryIntendedUses)

        if self.primaryIntendedUsers is not None and not isinstance(self.primaryIntendedUsers, str):
            self.primaryIntendedUsers = str(self.primaryIntendedUsers)

        if self.outOfScopeUseCases is not None and not isinstance(self.outOfScopeUseCases, str):
            self.outOfScopeUseCases = str(self.outOfScopeUseCases)

        if self.relevantFactors is not None and not isinstance(self.relevantFactors, str):
            self.relevantFactors = str(self.relevantFactors)

        if self.evaluationFactors is not None and not isinstance(self.evaluationFactors, str):
            self.evaluationFactors = str(self.evaluationFactors)

        if self.modelPerformanceMeasures is not None and not isinstance(self.modelPerformanceMeasures, str):
            self.modelPerformanceMeasures = str(self.modelPerformanceMeasures)

        if self.decisionThresholds is not None and not isinstance(self.decisionThresholds, str):
            self.decisionThresholds = str(self.decisionThresholds)

        if self.variationApproaches is not None and not isinstance(self.variationApproaches, str):
            self.variationApproaches = str(self.variationApproaches)

        if self.evaluationDatasets is not None and not isinstance(self.evaluationDatasets, str):
            self.evaluationDatasets = str(self.evaluationDatasets)

        if self.evaluationMotivation is not None and not isinstance(self.evaluationMotivation, str):
            self.evaluationMotivation = str(self.evaluationMotivation)

        if self.evaluationPreprocessing is not None and not isinstance(self.evaluationPreprocessing, str):
            self.evaluationPreprocessing = str(self.evaluationPreprocessing)

        if self.trainingDatasets is not None and not isinstance(self.trainingDatasets, str):
            self.trainingDatasets = str(self.trainingDatasets)

        if self.trainingMotivation is not None and not isinstance(self.trainingMotivation, str):
            self.trainingMotivation = str(self.trainingMotivation)

        if self.trainingPreprocessing is not None and not isinstance(self.trainingPreprocessing, str):
            self.trainingPreprocessing = str(self.trainingPreprocessing)

        if self.unitaryResults is not None and not isinstance(self.unitaryResults, str):
            self.unitaryResults = str(self.unitaryResults)

        if self.intersectionalResults is not None and not isinstance(self.intersectionalResults, str):
            self.intersectionalResults = str(self.intersectionalResults)

        if self.ethicalConsiderations is not None and not isinstance(self.ethicalConsiderations, str):
            self.ethicalConsiderations = str(self.ethicalConsiderations)

        if self.caveatsAndRecommendations is not None and not isinstance(self.caveatsAndRecommendations, str):
            self.caveatsAndRecommendations = str(self.caveatsAndRecommendations)

        super().__post_init__(**kwargs)


@dataclass
class ModelSpec(YAMLRoot):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = PERSONINFO.ModelSpec
    class_class_curie: ClassVar[str] = "personinfo:ModelSpec"
    class_name: ClassVar[str] = "ModelSpec"
    class_model_uri: ClassVar[URIRef] = PERSONINFO.ModelSpec

    dockerImage: str = None
    singularityImage: str = None
    repoUrl: Optional[str] = None
    committish: Optional[str] = None
    repoDownload: Optional[str] = None
    repoDownloadLocation: Optional[str] = None
    command: Optional[str] = None
    n_files: Optional[int] = None
    on_files: Optional[int] = None
    prediction_script: Optional[str] = None
    total: Optional[int] = None
    train: Optional[int] = None
    evaluate: Optional[int] = None
    test: Optional[int] = None
    male: Optional[Union[bool, Bool]] = None
    female: Optional[Union[bool, Bool]] = None
    age_histogram: Optional[str] = None
    race: Optional[str] = None
    imaging_contrast_info: Optional[str] = None
    dataset_sources: Optional[str] = None
    number_of_sites: Optional[int] = None
    sites: Optional[str] = None
    scanner_models: Optional[str] = None
    hardware: Optional[str] = None
    input_shape: Optional[str] = None
    block_shape: Optional[str] = None
    n_classes: Optional[int] = None
    lr: Optional[str] = None
    n_epochs: Optional[int] = None
    total_batch_size: Optional[int] = None
    number_of_gpus: Optional[int] = None
    loss_function: Optional[str] = None
    metrics: Optional[str] = None
    data_preprocessing: Optional[str] = None
    data_augmentation: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.dockerImage):
            self.MissingRequiredField("dockerImage")
        if not isinstance(self.dockerImage, str):
            self.dockerImage = str(self.dockerImage)

        if self._is_empty(self.singularityImage):
            self.MissingRequiredField("singularityImage")
        if not isinstance(self.singularityImage, str):
            self.singularityImage = str(self.singularityImage)

        if self.repoUrl is not None and not isinstance(self.repoUrl, str):
            self.repoUrl = str(self.repoUrl)

        if self.committish is not None and not isinstance(self.committish, str):
            self.committish = str(self.committish)

        if self.repoDownload is not None and not isinstance(self.repoDownload, str):
            self.repoDownload = str(self.repoDownload)

        if self.repoDownloadLocation is not None and not isinstance(self.repoDownloadLocation, str):
            self.repoDownloadLocation = str(self.repoDownloadLocation)

        if self.command is not None and not isinstance(self.command, str):
            self.command = str(self.command)

        if self.n_files is not None and not isinstance(self.n_files, int):
            self.n_files = int(self.n_files)

        if self.on_files is not None and not isinstance(self.on_files, int):
            self.on_files = int(self.on_files)

        if self.prediction_script is not None and not isinstance(self.prediction_script, str):
            self.prediction_script = str(self.prediction_script)

        if self.total is not None and not isinstance(self.total, int):
            self.total = int(self.total)

        if self.train is not None and not isinstance(self.train, int):
            self.train = int(self.train)

        if self.evaluate is not None and not isinstance(self.evaluate, int):
            self.evaluate = int(self.evaluate)

        if self.test is not None and not isinstance(self.test, int):
            self.test = int(self.test)

        if self.male is not None and not isinstance(self.male, Bool):
            self.male = Bool(self.male)

        if self.female is not None and not isinstance(self.female, Bool):
            self.female = Bool(self.female)

        if self.age_histogram is not None and not isinstance(self.age_histogram, str):
            self.age_histogram = str(self.age_histogram)

        if self.race is not None and not isinstance(self.race, str):
            self.race = str(self.race)

        if self.imaging_contrast_info is not None and not isinstance(self.imaging_contrast_info, str):
            self.imaging_contrast_info = str(self.imaging_contrast_info)

        if self.dataset_sources is not None and not isinstance(self.dataset_sources, str):
            self.dataset_sources = str(self.dataset_sources)

        if self.number_of_sites is not None and not isinstance(self.number_of_sites, int):
            self.number_of_sites = int(self.number_of_sites)

        if self.sites is not None and not isinstance(self.sites, str):
            self.sites = str(self.sites)

        if self.scanner_models is not None and not isinstance(self.scanner_models, str):
            self.scanner_models = str(self.scanner_models)

        if self.hardware is not None and not isinstance(self.hardware, str):
            self.hardware = str(self.hardware)

        if self.input_shape is not None and not isinstance(self.input_shape, str):
            self.input_shape = str(self.input_shape)

        if self.block_shape is not None and not isinstance(self.block_shape, str):
            self.block_shape = str(self.block_shape)

        if self.n_classes is not None and not isinstance(self.n_classes, int):
            self.n_classes = int(self.n_classes)

        if self.lr is not None and not isinstance(self.lr, str):
            self.lr = str(self.lr)

        if self.n_epochs is not None and not isinstance(self.n_epochs, int):
            self.n_epochs = int(self.n_epochs)

        if self.total_batch_size is not None and not isinstance(self.total_batch_size, int):
            self.total_batch_size = int(self.total_batch_size)

        if self.number_of_gpus is not None and not isinstance(self.number_of_gpus, int):
            self.number_of_gpus = int(self.number_of_gpus)

        if self.loss_function is not None and not isinstance(self.loss_function, str):
            self.loss_function = str(self.loss_function)

        if self.metrics is not None and not isinstance(self.metrics, str):
            self.metrics = str(self.metrics)

        if self.data_preprocessing is not None and not isinstance(self.data_preprocessing, str):
            self.data_preprocessing = str(self.data_preprocessing)

        if self.data_augmentation is not None and not isinstance(self.data_augmentation, str):
            self.data_augmentation = str(self.data_augmentation)

        super().__post_init__(**kwargs)


@dataclass
class Container(YAMLRoot):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = PERSONINFO.Container
    class_class_curie: ClassVar[str] = "personinfo:Container"
    class_name: ClassVar[str] = "Container"
    class_model_uri: ClassVar[URIRef] = PERSONINFO.Container

    modelcards: Optional[Union[Union[dict, ModelCard], List[Union[dict, ModelCard]]]] = empty_list()
    modelSpecs: Optional[Union[Union[dict, ModelSpec], List[Union[dict, ModelSpec]]]] = empty_list()

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if not isinstance(self.modelcards, list):
            self.modelcards = [self.modelcards] if self.modelcards is not None else []
        self.modelcards = [v if isinstance(v, ModelCard) else ModelCard(**as_dict(v)) for v in self.modelcards]

        if not isinstance(self.modelSpecs, list):
            self.modelSpecs = [self.modelSpecs] if self.modelSpecs is not None else []
        self.modelSpecs = [v if isinstance(v, ModelSpec) else ModelSpec(**as_dict(v)) for v in self.modelSpecs]

        super().__post_init__(**kwargs)


# Enumerations


# Slots
class slots:
    pass

slots.modelCard__organization = Slot(uri=PERSONINFO.organization, name="modelCard__organization", curie=PERSONINFO.curie('organization'),
                   model_uri=PERSONINFO.modelCard__organization, domain=None, range=Optional[str])

slots.modelCard__modelDate = Slot(uri=PERSONINFO.modelDate, name="modelCard__modelDate", curie=PERSONINFO.curie('modelDate'),
                   model_uri=PERSONINFO.modelCard__modelDate, domain=None, range=Optional[str])

slots.modelCard__modelVersion = Slot(uri=PERSONINFO.modelVersion, name="modelCard__modelVersion", curie=PERSONINFO.curie('modelVersion'),
                   model_uri=PERSONINFO.modelCard__modelVersion, domain=None, range=Optional[float])

slots.modelCard__modelType = Slot(uri=PERSONINFO.modelType, name="modelCard__modelType", curie=PERSONINFO.curie('modelType'),
                   model_uri=PERSONINFO.modelCard__modelType, domain=None, range=Optional[str])

slots.modelCard__trainingApproaches = Slot(uri=PERSONINFO.trainingApproaches, name="modelCard__trainingApproaches", curie=PERSONINFO.curie('trainingApproaches'),
                   model_uri=PERSONINFO.modelCard__trainingApproaches, domain=None, range=Optional[str])

slots.modelCard__moreInformation = Slot(uri=PERSONINFO.moreInformation, name="modelCard__moreInformation", curie=PERSONINFO.curie('moreInformation'),
                   model_uri=PERSONINFO.modelCard__moreInformation, domain=None, range=Optional[str])

slots.modelCard__citationDetails = Slot(uri=PERSONINFO.citationDetails, name="modelCard__citationDetails", curie=PERSONINFO.curie('citationDetails'),
                   model_uri=PERSONINFO.modelCard__citationDetails, domain=None, range=Optional[str])

slots.modelCard__contactInfo = Slot(uri=PERSONINFO.contactInfo, name="modelCard__contactInfo", curie=PERSONINFO.curie('contactInfo'),
                   model_uri=PERSONINFO.modelCard__contactInfo, domain=None, range=Optional[str])

slots.modelCard__primaryIntendedUses = Slot(uri=PERSONINFO.primaryIntendedUses, name="modelCard__primaryIntendedUses", curie=PERSONINFO.curie('primaryIntendedUses'),
                   model_uri=PERSONINFO.modelCard__primaryIntendedUses, domain=None, range=Optional[str])

slots.modelCard__primaryIntendedUsers = Slot(uri=PERSONINFO.primaryIntendedUsers, name="modelCard__primaryIntendedUsers", curie=PERSONINFO.curie('primaryIntendedUsers'),
                   model_uri=PERSONINFO.modelCard__primaryIntendedUsers, domain=None, range=Optional[str])

slots.modelCard__outOfScopeUseCases = Slot(uri=PERSONINFO.outOfScopeUseCases, name="modelCard__outOfScopeUseCases", curie=PERSONINFO.curie('outOfScopeUseCases'),
                   model_uri=PERSONINFO.modelCard__outOfScopeUseCases, domain=None, range=Optional[str])

slots.modelCard__relevantFactors = Slot(uri=PERSONINFO.relevantFactors, name="modelCard__relevantFactors", curie=PERSONINFO.curie('relevantFactors'),
                   model_uri=PERSONINFO.modelCard__relevantFactors, domain=None, range=Optional[str])

slots.modelCard__evaluationFactors = Slot(uri=PERSONINFO.evaluationFactors, name="modelCard__evaluationFactors", curie=PERSONINFO.curie('evaluationFactors'),
                   model_uri=PERSONINFO.modelCard__evaluationFactors, domain=None, range=Optional[str])

slots.modelCard__modelPerformanceMeasures = Slot(uri=PERSONINFO.modelPerformanceMeasures, name="modelCard__modelPerformanceMeasures", curie=PERSONINFO.curie('modelPerformanceMeasures'),
                   model_uri=PERSONINFO.modelCard__modelPerformanceMeasures, domain=None, range=Optional[str])

slots.modelCard__decisionThresholds = Slot(uri=PERSONINFO.decisionThresholds, name="modelCard__decisionThresholds", curie=PERSONINFO.curie('decisionThresholds'),
                   model_uri=PERSONINFO.modelCard__decisionThresholds, domain=None, range=Optional[str])

slots.modelCard__variationApproaches = Slot(uri=PERSONINFO.variationApproaches, name="modelCard__variationApproaches", curie=PERSONINFO.curie('variationApproaches'),
                   model_uri=PERSONINFO.modelCard__variationApproaches, domain=None, range=Optional[str])

slots.modelCard__evaluationDatasets = Slot(uri=PERSONINFO.evaluationDatasets, name="modelCard__evaluationDatasets", curie=PERSONINFO.curie('evaluationDatasets'),
                   model_uri=PERSONINFO.modelCard__evaluationDatasets, domain=None, range=Optional[str])

slots.modelCard__evaluationMotivation = Slot(uri=PERSONINFO.evaluationMotivation, name="modelCard__evaluationMotivation", curie=PERSONINFO.curie('evaluationMotivation'),
                   model_uri=PERSONINFO.modelCard__evaluationMotivation, domain=None, range=Optional[str])

slots.modelCard__evaluationPreprocessing = Slot(uri=PERSONINFO.evaluationPreprocessing, name="modelCard__evaluationPreprocessing", curie=PERSONINFO.curie('evaluationPreprocessing'),
                   model_uri=PERSONINFO.modelCard__evaluationPreprocessing, domain=None, range=Optional[str])

slots.modelCard__trainingDatasets = Slot(uri=PERSONINFO.trainingDatasets, name="modelCard__trainingDatasets", curie=PERSONINFO.curie('trainingDatasets'),
                   model_uri=PERSONINFO.modelCard__trainingDatasets, domain=None, range=Optional[str])

slots.modelCard__trainingMotivation = Slot(uri=PERSONINFO.trainingMotivation, name="modelCard__trainingMotivation", curie=PERSONINFO.curie('trainingMotivation'),
                   model_uri=PERSONINFO.modelCard__trainingMotivation, domain=None, range=Optional[str])

slots.modelCard__trainingPreprocessing = Slot(uri=PERSONINFO.trainingPreprocessing, name="modelCard__trainingPreprocessing", curie=PERSONINFO.curie('trainingPreprocessing'),
                   model_uri=PERSONINFO.modelCard__trainingPreprocessing, domain=None, range=Optional[str])

slots.modelCard__unitaryResults = Slot(uri=PERSONINFO.unitaryResults, name="modelCard__unitaryResults", curie=PERSONINFO.curie('unitaryResults'),
                   model_uri=PERSONINFO.modelCard__unitaryResults, domain=None, range=Optional[str])

slots.modelCard__intersectionalResults = Slot(uri=PERSONINFO.intersectionalResults, name="modelCard__intersectionalResults", curie=PERSONINFO.curie('intersectionalResults'),
                   model_uri=PERSONINFO.modelCard__intersectionalResults, domain=None, range=Optional[str])

slots.modelCard__ethicalConsiderations = Slot(uri=PERSONINFO.ethicalConsiderations, name="modelCard__ethicalConsiderations", curie=PERSONINFO.curie('ethicalConsiderations'),
                   model_uri=PERSONINFO.modelCard__ethicalConsiderations, domain=None, range=Optional[str])

slots.modelCard__caveatsAndRecommendations = Slot(uri=PERSONINFO.caveatsAndRecommendations, name="modelCard__caveatsAndRecommendations", curie=PERSONINFO.curie('caveatsAndRecommendations'),
                   model_uri=PERSONINFO.modelCard__caveatsAndRecommendations, domain=None, range=Optional[str])

slots.modelSpec__dockerImage = Slot(uri=PERSONINFO.dockerImage, name="modelSpec__dockerImage", curie=PERSONINFO.curie('dockerImage'),
                   model_uri=PERSONINFO.modelSpec__dockerImage, domain=None, range=str)

slots.modelSpec__singularityImage = Slot(uri=PERSONINFO.singularityImage, name="modelSpec__singularityImage", curie=PERSONINFO.curie('singularityImage'),
                   model_uri=PERSONINFO.modelSpec__singularityImage, domain=None, range=str)

slots.modelSpec__repoUrl = Slot(uri=PERSONINFO.repoUrl, name="modelSpec__repoUrl", curie=PERSONINFO.curie('repoUrl'),
                   model_uri=PERSONINFO.modelSpec__repoUrl, domain=None, range=Optional[str])

slots.modelSpec__committish = Slot(uri=PERSONINFO.committish, name="modelSpec__committish", curie=PERSONINFO.curie('committish'),
                   model_uri=PERSONINFO.modelSpec__committish, domain=None, range=Optional[str])

slots.modelSpec__repoDownload = Slot(uri=PERSONINFO.repoDownload, name="modelSpec__repoDownload", curie=PERSONINFO.curie('repoDownload'),
                   model_uri=PERSONINFO.modelSpec__repoDownload, domain=None, range=Optional[str])

slots.modelSpec__repoDownloadLocation = Slot(uri=PERSONINFO.repoDownloadLocation, name="modelSpec__repoDownloadLocation", curie=PERSONINFO.curie('repoDownloadLocation'),
                   model_uri=PERSONINFO.modelSpec__repoDownloadLocation, domain=None, range=Optional[str])

slots.modelSpec__command = Slot(uri=PERSONINFO.command, name="modelSpec__command", curie=PERSONINFO.curie('command'),
                   model_uri=PERSONINFO.modelSpec__command, domain=None, range=Optional[str])

slots.modelSpec__n_files = Slot(uri=PERSONINFO.n_files, name="modelSpec__n_files", curie=PERSONINFO.curie('n_files'),
                   model_uri=PERSONINFO.modelSpec__n_files, domain=None, range=Optional[int])

slots.modelSpec__on_files = Slot(uri=PERSONINFO.on_files, name="modelSpec__on_files", curie=PERSONINFO.curie('on_files'),
                   model_uri=PERSONINFO.modelSpec__on_files, domain=None, range=Optional[int])

slots.modelSpec__prediction_script = Slot(uri=PERSONINFO.prediction_script, name="modelSpec__prediction_script", curie=PERSONINFO.curie('prediction_script'),
                   model_uri=PERSONINFO.modelSpec__prediction_script, domain=None, range=Optional[str])

slots.modelSpec__total = Slot(uri=PERSONINFO.total, name="modelSpec__total", curie=PERSONINFO.curie('total'),
                   model_uri=PERSONINFO.modelSpec__total, domain=None, range=Optional[int])

slots.modelSpec__train = Slot(uri=PERSONINFO.train, name="modelSpec__train", curie=PERSONINFO.curie('train'),
                   model_uri=PERSONINFO.modelSpec__train, domain=None, range=Optional[int])

slots.modelSpec__evaluate = Slot(uri=PERSONINFO.evaluate, name="modelSpec__evaluate", curie=PERSONINFO.curie('evaluate'),
                   model_uri=PERSONINFO.modelSpec__evaluate, domain=None, range=Optional[int])

slots.modelSpec__test = Slot(uri=PERSONINFO.test, name="modelSpec__test", curie=PERSONINFO.curie('test'),
                   model_uri=PERSONINFO.modelSpec__test, domain=None, range=Optional[int])

slots.modelSpec__male = Slot(uri=PERSONINFO.male, name="modelSpec__male", curie=PERSONINFO.curie('male'),
                   model_uri=PERSONINFO.modelSpec__male, domain=None, range=Optional[Union[bool, Bool]])

slots.modelSpec__female = Slot(uri=PERSONINFO.female, name="modelSpec__female", curie=PERSONINFO.curie('female'),
                   model_uri=PERSONINFO.modelSpec__female, domain=None, range=Optional[Union[bool, Bool]])

slots.modelSpec__age_histogram = Slot(uri=PERSONINFO.age_histogram, name="modelSpec__age_histogram", curie=PERSONINFO.curie('age_histogram'),
                   model_uri=PERSONINFO.modelSpec__age_histogram, domain=None, range=Optional[str])

slots.modelSpec__race = Slot(uri=PERSONINFO.race, name="modelSpec__race", curie=PERSONINFO.curie('race'),
                   model_uri=PERSONINFO.modelSpec__race, domain=None, range=Optional[str])

slots.modelSpec__imaging_contrast_info = Slot(uri=PERSONINFO.imaging_contrast_info, name="modelSpec__imaging_contrast_info", curie=PERSONINFO.curie('imaging_contrast_info'),
                   model_uri=PERSONINFO.modelSpec__imaging_contrast_info, domain=None, range=Optional[str])

slots.modelSpec__dataset_sources = Slot(uri=PERSONINFO.dataset_sources, name="modelSpec__dataset_sources", curie=PERSONINFO.curie('dataset_sources'),
                   model_uri=PERSONINFO.modelSpec__dataset_sources, domain=None, range=Optional[str])

slots.modelSpec__number_of_sites = Slot(uri=PERSONINFO.number_of_sites, name="modelSpec__number_of_sites", curie=PERSONINFO.curie('number_of_sites'),
                   model_uri=PERSONINFO.modelSpec__number_of_sites, domain=None, range=Optional[int])

slots.modelSpec__sites = Slot(uri=PERSONINFO.sites, name="modelSpec__sites", curie=PERSONINFO.curie('sites'),
                   model_uri=PERSONINFO.modelSpec__sites, domain=None, range=Optional[str])

slots.modelSpec__scanner_models = Slot(uri=PERSONINFO.scanner_models, name="modelSpec__scanner_models", curie=PERSONINFO.curie('scanner_models'),
                   model_uri=PERSONINFO.modelSpec__scanner_models, domain=None, range=Optional[str])

slots.modelSpec__hardware = Slot(uri=PERSONINFO.hardware, name="modelSpec__hardware", curie=PERSONINFO.curie('hardware'),
                   model_uri=PERSONINFO.modelSpec__hardware, domain=None, range=Optional[str])

slots.modelSpec__input_shape = Slot(uri=PERSONINFO.input_shape, name="modelSpec__input_shape", curie=PERSONINFO.curie('input_shape'),
                   model_uri=PERSONINFO.modelSpec__input_shape, domain=None, range=Optional[str])

slots.modelSpec__block_shape = Slot(uri=PERSONINFO.block_shape, name="modelSpec__block_shape", curie=PERSONINFO.curie('block_shape'),
                   model_uri=PERSONINFO.modelSpec__block_shape, domain=None, range=Optional[str])

slots.modelSpec__n_classes = Slot(uri=PERSONINFO.n_classes, name="modelSpec__n_classes", curie=PERSONINFO.curie('n_classes'),
                   model_uri=PERSONINFO.modelSpec__n_classes, domain=None, range=Optional[int])

slots.modelSpec__lr = Slot(uri=PERSONINFO.lr, name="modelSpec__lr", curie=PERSONINFO.curie('lr'),
                   model_uri=PERSONINFO.modelSpec__lr, domain=None, range=Optional[str])

slots.modelSpec__n_epochs = Slot(uri=PERSONINFO.n_epochs, name="modelSpec__n_epochs", curie=PERSONINFO.curie('n_epochs'),
                   model_uri=PERSONINFO.modelSpec__n_epochs, domain=None, range=Optional[int])

slots.modelSpec__total_batch_size = Slot(uri=PERSONINFO.total_batch_size, name="modelSpec__total_batch_size", curie=PERSONINFO.curie('total_batch_size'),
                   model_uri=PERSONINFO.modelSpec__total_batch_size, domain=None, range=Optional[int])

slots.modelSpec__number_of_gpus = Slot(uri=PERSONINFO.number_of_gpus, name="modelSpec__number_of_gpus", curie=PERSONINFO.curie('number_of_gpus'),
                   model_uri=PERSONINFO.modelSpec__number_of_gpus, domain=None, range=Optional[int])

slots.modelSpec__loss_function = Slot(uri=PERSONINFO.loss_function, name="modelSpec__loss_function", curie=PERSONINFO.curie('loss_function'),
                   model_uri=PERSONINFO.modelSpec__loss_function, domain=None, range=Optional[str])

slots.modelSpec__metrics = Slot(uri=PERSONINFO.metrics, name="modelSpec__metrics", curie=PERSONINFO.curie('metrics'),
                   model_uri=PERSONINFO.modelSpec__metrics, domain=None, range=Optional[str])

slots.modelSpec__data_preprocessing = Slot(uri=PERSONINFO.data_preprocessing, name="modelSpec__data_preprocessing", curie=PERSONINFO.curie('data_preprocessing'),
                   model_uri=PERSONINFO.modelSpec__data_preprocessing, domain=None, range=Optional[str])

slots.modelSpec__data_augmentation = Slot(uri=PERSONINFO.data_augmentation, name="modelSpec__data_augmentation", curie=PERSONINFO.curie('data_augmentation'),
                   model_uri=PERSONINFO.modelSpec__data_augmentation, domain=None, range=Optional[str])

slots.container__modelcards = Slot(uri=PERSONINFO.modelcards, name="container__modelcards", curie=PERSONINFO.curie('modelcards'),
                   model_uri=PERSONINFO.container__modelcards, domain=None, range=Optional[Union[Union[dict, ModelCard], List[Union[dict, ModelCard]]]])

slots.container__modelSpecs = Slot(uri=PERSONINFO.modelSpecs, name="container__modelSpecs", curie=PERSONINFO.curie('modelSpecs'),
                   model_uri=PERSONINFO.container__modelSpecs, domain=None, range=Optional[Union[Union[dict, ModelSpec], List[Union[dict, ModelSpec]]]])
