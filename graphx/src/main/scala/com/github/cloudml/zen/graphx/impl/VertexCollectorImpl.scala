/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.cloudml.zen.graphx.impl

import com.github.cloudml.zen.graphx._
import com.github.cloudml.zen.graphx.util.{PSUtils => GPSUtils}
import org.parameterserver.client.PSClient

import scala.reflect.ClassTag

class VertexCollectorImpl[VD: ClassTag, ED: ClassTag](
  val psClient: PSClient,
  val psName: String) extends VertexCollector[VD, ED] {
  override def update(vid: VertexId, msg: VD): Unit = {
    GPSUtils.batchUpadte(psClient, psName, Array(vid.toInt), Array(msg))
  }

  override def inc(vid: VertexId, msg: VD): Unit = {
    GPSUtils.batchInc(psClient, psName, Array(vid.toInt), Array(msg))
  }
}
